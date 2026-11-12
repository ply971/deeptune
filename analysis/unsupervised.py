"""Unsupervised analysis of clip embeddings: which videos look alike, which are unusual.

Input: an embedding table, as ``deeptune.py``'s embed step writes it
(``embed_output_*/*_embeddings.parquet``: one column per embedding dimension,
named ``0, 1, 2, ...``, then ``label`` and any passthrough columns), or any
CSV/parquet whose feature columns are given with ``--feature_prefix``.

What it computes (scikit-learn / SciPy, plus a small torch autoencoder):

- **PCA**: explained variance per component and cumulative; 2-D projection.
- **LDA** (when labels exist): the projection that best separates the labelled
  classes, and its explained-variance ratio.
- **K-Means** (k-means++, ``n_init`` restarts) for every k in a range: inertia
  (elbow) and silhouette; the suggested k is the best silhouette. The elbow is
  reported too (the point furthest from the line joining the first and last
  inertia), since the two can disagree.
- **Agglomerative** clustering (Ward / average / complete / single linkage) at
  the suggested k, with the full SciPy linkage matrix (dendrogram data).
- **ARI / NMI** of each clustering against the labels, when labels exist.
- **Anomaly detection**: an autoencoder fitted on the ``fit`` rows; a row's
  anomaly score is its reconstruction error; the threshold is the
  ``1 - target_fpr`` quantile of the ``val`` rows' scores, so about
  ``target_fpr`` of normal clips are flagged. With an ``--outlier_label``, AUC
  and the observed false-positive / detection rates are reported.

Features are standardised (z-score with the fit rows' mean and std) before
everything. Outputs go to ``--out``: ``summary.json``, ``pca.csv``,
``kmeans_sweep.csv``, ``clusters.csv``, ``linkage.npy``, ``anomaly.csv`` and
PNG plots (``pca.png``, ``lda.png``, ``kmeans_sweep.png``, ``dendrogram.png``,
``anomaly.png``).

    python -m analysis.unsupervised --embeddings run/embed_output_x/x_cls_embeddings.parquet --out run/unsupervised
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

LINKAGES = ('ward', 'average', 'complete', 'single')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def feature_columns(frame: pd.DataFrame, prefix: Optional[str] = None) -> list:
    """Embedding columns: names that are integers (DeepTune's embed output), or names starting with ``prefix``."""
    if prefix:
        cols = [c for c in frame.columns if str(c).startswith(prefix)]
    else:
        cols = [c for c in frame.columns if str(c).strip().lstrip('-').isdigit()]
    if not cols:
        raise ValueError('No embedding columns found: expected columns named 0, 1, 2, ... '
                         '(DeepTune embed output) or pass --feature_prefix.')
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if bad:
        raise ValueError(f'Embedding columns must be numeric: {bad[:5]}')
    return cols


def load_embeddings(path, label_column: Optional[str] = 'label', prefix: Optional[str] = None):
    """(X [N, D] float64, labels or None, the other columns)."""
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == '.csv':
        frame = pd.read_csv(path)
    else:
        raise ValueError(f'{path}: expected a .parquet or .csv embedding table.')
    cols = feature_columns(frame, prefix)
    X = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f'{path}: the embeddings contain NaN or infinite values.')
    labels = None
    if label_column and label_column in frame.columns:
        labels = frame[label_column].astype(str).to_numpy()
    rest = frame.drop(columns=cols)
    return X, labels, rest


def standardise(X: np.ndarray, fit_rows: Optional[np.ndarray] = None) -> np.ndarray:
    ref = X if fit_rows is None else X[fit_rows]
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    return (X - mean) / np.where(std > 1e-12, std, 1.0)


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def pca_analysis(X: np.ndarray, n_components: Optional[int] = None, seed: int = 0) -> dict:
    from sklearn.decomposition import PCA
    n = min(X.shape) if n_components is None else min(n_components, *X.shape)
    pca = PCA(n_components=n, random_state=seed).fit(X)
    ratio = pca.explained_variance_ratio_
    return {'projection': pca.transform(X), 'explained_variance_ratio': ratio,
            'cumulative': np.cumsum(ratio),
            'components_for_90pct': int(np.searchsorted(np.cumsum(ratio), 0.9 - 1e-12) + 1) if ratio.sum() >= 0.9 - 1e-9 else None}


def lda_analysis(X: np.ndarray, labels: Sequence) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError('LDA needs at least two classes.')
    lda = LinearDiscriminantAnalysis(solver='svd').fit(X, labels)
    return {'projection': lda.transform(X), 'explained_variance_ratio': lda.explained_variance_ratio_,
            'classes': classes.tolist(), 'training_accuracy': float(lda.score(X, labels))}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def elbow_k(ks: Sequence[int], inertia: Sequence[float]) -> int:
    """The k furthest from the straight line joining the first and last (k, inertia) points (both axes scaled to [0, 1])."""
    ks = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)
    if len(ks) < 3:
        return int(ks[0])
    xn = (ks - ks[0]) / (ks[-1] - ks[0])
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    # distance from (xn, yn) to the line through (0, yn[0]) and (1, yn[-1])
    dx, dy = 1.0, yn[-1] - yn[0]
    dist = np.abs(dy * xn - dx * (yn - yn[0])) / np.hypot(dx, dy)
    return int(ks[int(np.argmax(dist))])


def kmeans_sweep(X: np.ndarray, ks: Sequence[int], n_init: int = 10, seed: int = 0) -> pd.DataFrame:
    """Inertia and silhouette for each k (k-means++ with ``n_init`` restarts)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows = []
    for k in ks:
        if not 2 <= k < len(X):
            raise ValueError(f'k={k}: need 2 <= k < number of rows ({len(X)}).')
        km = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=seed).fit(X)
        rows.append({'k': int(k), 'inertia': float(km.inertia_), 'silhouette': float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(rows)


def kmeans(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=seed).fit_predict(X)


def agglomerative(X: np.ndarray, k: int, linkage: str = 'ward') -> tuple[np.ndarray, np.ndarray]:
    """(cluster per row, SciPy linkage matrix [N-1, 4]) - the matrix is the dendrogram."""
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    if linkage not in LINKAGES:
        raise ValueError(f'linkage must be one of {LINKAGES}')
    Z = scipy_linkage(X, method=linkage, metric='euclidean')
    clusters = fcluster(Z, t=k, criterion='maxclust') - 1
    return clusters, Z


def agreement(labels: Sequence, clusters: Sequence) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return {'ari': float(adjusted_rand_score(labels, clusters)), 'nmi': float(normalized_mutual_info_score(labels, clusters))}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderSettings:
    hidden: int = 64
    bottleneck: int = 8
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    seed: int = 0
    device: str = 'cpu'


def fit_autoencoder(X: np.ndarray, settings: AutoencoderSettings = AutoencoderSettings()):
    """A small MLP autoencoder D -> hidden -> bottleneck -> hidden -> D (MSE loss)."""
    import torch
    from torch import nn
    gen = torch.Generator().manual_seed(settings.seed)
    torch.manual_seed(settings.seed)
    d = X.shape[1]
    bottleneck = max(1, min(settings.bottleneck, d))
    model = nn.Sequential(nn.Linear(d, settings.hidden), nn.ReLU(), nn.Linear(settings.hidden, bottleneck), nn.ReLU(),
                          nn.Linear(bottleneck, settings.hidden), nn.ReLU(), nn.Linear(settings.hidden, d)).to(settings.device)
    opt = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    data = torch.as_tensor(X, dtype=torch.float32, device=settings.device)
    losses = []
    model.train()
    for _ in range(settings.epochs):
        order = torch.randperm(len(data), generator=gen).to(settings.device)
        total = 0.0
        for i in range(0, len(data), settings.batch_size):
            batch = data[order[i:i + settings.batch_size]]
            loss = ((model(batch) - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    model.eval()
    model.losses = losses
    return model


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        data = torch.as_tensor(X, dtype=torch.float32, device=device)
        return ((model(data) - data) ** 2).mean(dim=1).cpu().numpy().astype(np.float64)


def fpr_threshold(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Score above which about ``target_fpr`` of normal rows lie (their ``1 - target_fpr`` quantile)."""
    if not 0 < target_fpr < 1:
        raise ValueError('target_fpr must be between 0 and 1')
    if len(normal_scores) == 0:
        raise ValueError('need validation rows to set the threshold')
    return float(np.quantile(np.asarray(normal_scores, dtype=np.float64), 1 - target_fpr, method='higher'))


def auroc(scores: np.ndarray, positive: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return None
    return float(roc_auc_score(positive, scores))


def assign_roles(n: int, fit_fraction: float = 0.6, val_fraction: float = 0.2, seed: int = 0,
                 exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """'fit' / 'val' / 'test' per row, at random; rows in ``exclude`` (known outliers) are always 'test'."""
    if not (0 < fit_fraction < 1 and 0 < val_fraction < 1 and fit_fraction + val_fraction < 1 + 1e-9):
        raise ValueError('fit and val fractions must be in (0, 1) and sum to at most 1')
    roles = np.array(['test'] * n, dtype=object)
    pool = np.flatnonzero(~exclude) if exclude is not None else np.arange(n)
    order = np.random.default_rng(seed).permutation(pool)
    n_fit = int(round(fit_fraction * len(order)))
    n_val = int(round(val_fraction * len(order)))
    roles[order[:n_fit]] = 'fit'
    roles[order[n_fit:n_fit + n_val]] = 'val'
    return roles


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------

@dataclass
class AnalysisSettings:
    ks: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)
    n_init: int = 10
    linkage: str = 'ward'
    pca_components: Optional[int] = None
    target_fpr: float = 0.05
    fit_fraction: float = 0.6
    val_fraction: float = 0.2
    outlier_labels: Sequence[str] = ()
    role_column: Optional[str] = None
    seed: int = 0
    plots: bool = True
    autoencoder: AutoencoderSettings = field(default_factory=AutoencoderSettings)


def analyse(X: np.ndarray, labels: Optional[np.ndarray], out_dir, settings: AnalysisSettings = AnalysisSettings(),
            roles: Optional[np.ndarray] = None, extra: Optional[pd.DataFrame] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(X)
    if n < 4:
        raise ValueError('Unsupervised analysis needs at least 4 rows.')
    outlier = np.isin(labels, list(settings.outlier_labels)) if labels is not None and settings.outlier_labels else np.zeros(n, bool)
    if roles is None:
        roles = assign_roles(n, settings.fit_fraction, settings.val_fraction, settings.seed, outlier)
    roles = np.asarray(roles, dtype=object)
    if not set(roles) <= {'fit', 'val', 'test'}:
        raise ValueError("roles must be 'fit', 'val' or 'test'")
    if outlier[roles != 'test'].any():
        raise ValueError('rows with an --outlier_label must not be used to fit the autoencoder or set its threshold.')
    if (roles == 'fit').sum() < 2 or (roles == 'val').sum() < 1:
        raise ValueError('need at least 2 fit rows and 1 val row')
    Z = standardise(X, roles == 'fit')
    summary: dict = {'rows': int(n), 'dimensions': int(X.shape[1]), 'labels': None if labels is None else sorted(set(labels.tolist()))}

    pca = pca_analysis(Z, settings.pca_components, settings.seed)
    ratio = pca['explained_variance_ratio']
    summary['pca'] = {'explained_variance_ratio': [round(float(v), 6) for v in ratio[:20]],
                      'first_two': round(float(ratio[:2].sum()), 6), 'components_for_90pct': pca['components_for_90pct']}
    pd.DataFrame(pca['projection'][:, :min(10, pca['projection'].shape[1])],
                 columns=[f'pc{i + 1}' for i in range(min(10, pca['projection'].shape[1]))]).to_csv(out / 'pca.csv', index=False)

    lda = None
    clean = labels is not None and len(set(labels[~outlier].tolist())) >= 2
    if clean:
        lda = lda_analysis(Z[~outlier], labels[~outlier])
        summary['lda'] = {'explained_variance_ratio': [round(float(v), 6) for v in lda['explained_variance_ratio']],
                          'training_accuracy': round(lda['training_accuracy'], 6)}

    ks = [k for k in settings.ks if 2 <= k < n]
    if not ks:
        raise ValueError('no k in range: need 2 <= k < rows')
    sweep = kmeans_sweep(Z, ks, settings.n_init, settings.seed)
    sweep.to_csv(out / 'kmeans_sweep.csv', index=False)
    best_k = int(sweep.loc[sweep['silhouette'].idxmax(), 'k'])
    summary['kmeans'] = {'suggested_k_silhouette': best_k, 'elbow_k': elbow_k(sweep['k'], sweep['inertia']),
                         'sweep': sweep.round(6).to_dict(orient='records')}
    km = kmeans(Z, best_k, settings.n_init, settings.seed)
    agg, link = agglomerative(Z, best_k, settings.linkage)
    np.save(out / 'linkage.npy', link)
    summary['agglomerative'] = {'linkage': settings.linkage, 'k': best_k}
    if labels is not None:
        summary['kmeans']['agreement'] = agreement(labels, km)
        summary['agglomerative']['agreement'] = agreement(labels, agg)

    model = fit_autoencoder(Z[roles == 'fit'], settings.autoencoder)
    score = reconstruction_error(model, Z)
    threshold = fpr_threshold(score[roles == 'val'], settings.target_fpr)
    flagged = score > threshold
    anomaly = {'target_fpr': settings.target_fpr, 'threshold': threshold, 'flagged': int(flagged.sum()),
               'fit_rows': int((roles == 'fit').sum()), 'val_rows': int((roles == 'val').sum()),
               'test_rows': int((roles == 'test').sum()), 'final_training_loss': round(model.losses[-1], 6)}
    test = roles == 'test'
    normal_test = test & ~outlier
    if normal_test.any():
        anomaly['test_false_positive_rate'] = float(flagged[normal_test].mean())
    if outlier.any():
        anomaly['outlier_labels'] = list(settings.outlier_labels)
        anomaly['detection_rate'] = float(flagged[outlier].mean())
        anomaly['auc'] = auroc(score[test], outlier[test])
    summary['anomaly'] = anomaly

    table = pd.DataFrame({'role': roles, 'kmeans_cluster': km, 'agglomerative_cluster': agg,
                          'pc1': pca['projection'][:, 0], 'pc2': pca['projection'][:, 1] if pca['projection'].shape[1] > 1 else 0.0,
                          'anomaly_score': score, 'anomalous': flagged})
    if labels is not None:
        table.insert(0, 'label', labels)
    if extra is not None:
        keep = [c for c in extra.columns if c not in table.columns and c != 'label']
        table = pd.concat([extra[keep].reset_index(drop=True), table], axis=1)
    table.to_csv(out / 'clusters.csv', index=False)
    table[[c for c in table.columns if c in ('label', 'role', 'anomaly_score', 'anomalous') or c in (extra.columns if extra is not None else [])]].to_csv(out / 'anomaly.csv', index=False)

    if settings.plots:
        summary['plots'] = _plots(out, pca, lda, labels, outlier, sweep, link, score, roles, threshold)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    return summary


def _plots(out: Path, pca, lda, labels, outlier, sweep, link, score, roles, threshold) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    files = []

    def scatter(proj, labs, name, title):
        fig, ax = plt.subplots(figsize=(5, 4))
        y = proj[:, 1] if proj.shape[1] > 1 else np.zeros(len(proj))
        if labs is None:
            ax.scatter(proj[:, 0], y, s=10)
        else:
            for lab in sorted(set(labs.tolist())):
                m = labs == lab
                ax.scatter(proj[m, 0], y[m], s=10, label=str(lab))
            ax.legend(fontsize=7, loc='best')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        files.append(name)

    scatter(pca['projection'], labels, 'pca.png', 'PCA (first two components)')
    if lda is not None:
        scatter(lda['projection'], labels[~outlier], 'lda.png', 'LDA (labelled classes, outliers left out)')
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(sweep['k'], sweep['inertia'], 'o-', color='tab:blue')
    ax1.set_xlabel('k')
    ax1.set_ylabel('inertia', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(sweep['k'], sweep['silhouette'], 's--', color='tab:orange')
    ax2.set_ylabel('silhouette', color='tab:orange')
    ax1.set_title('K-Means: elbow and silhouette')
    fig.tight_layout()
    fig.savefig(out / 'kmeans_sweep.png', dpi=110)
    plt.close(fig)
    files.append('kmeans_sweep.png')
    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(link, ax=ax, no_labels=True, truncate_mode='lastp' if len(link) > 60 else None, p=30)
    ax.set_title('Agglomerative clustering (dendrogram)')
    fig.tight_layout()
    fig.savefig(out / 'dendrogram.png', dpi=110)
    plt.close(fig)
    files.append('dendrogram.png')
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.histogram_bin_edges(score, bins=30)
    for role in ('fit', 'val', 'test'):
        m = (roles == role) & ~outlier
        if m.any():
            ax.hist(score[m], bins=bins, alpha=0.5, label=f'{role} (normal)')
    if outlier.any():
        ax.hist(score[outlier], bins=bins, alpha=0.6, label='outlier label')
    ax.axvline(threshold, color='k', linestyle='--', label='threshold')
    ax.set_xlabel('reconstruction error')
    ax.legend(fontsize=7)
    ax.set_title('Autoencoder anomaly score')
    fig.tight_layout()
    fig.savefig(out / 'anomaly.png', dpi=110)
    plt.close(fig)
    files.append('anomaly.png')
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--embeddings', type=Path, required=True, help='Embedding table (.parquet or .csv).')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label_column', default='label', help="Labels column, if any (default 'label'; '' for none).")
    p.add_argument('--feature_prefix', default=None, help='Feature columns start with this (default: integer-named columns).')
    p.add_argument('--k_min', type=int, default=2)
    p.add_argument('--k_max', type=int, default=8)
    p.add_argument('--n_init', type=int, default=10, help='K-Means restarts (default 10).')
    p.add_argument('--linkage', choices=LINKAGES, default='ward')
    p.add_argument('--pca_components', type=int, default=None)
    p.add_argument('--target_fpr', type=float, default=0.05,
                   help='Share of normal clips the anomaly threshold may flag (default 0.05).')
    p.add_argument('--outlier_label', action='append', default=[],
                   help='Rows with this label are known outliers: never used to fit, scored for AUC. Repeatable.')
    p.add_argument('--role_column', default=None,
                   help="Column saying fit/val/test per row (else: random 60/20/20 split; 'train' counts as fit).")
    p.add_argument('--ae_epochs', type=int, default=200)
    p.add_argument('--ae_bottleneck', type=int, default=8)
    p.add_argument('--ae_hidden', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_plots', action='store_true')
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.embeddings.is_file():
        parser.error(f'--embeddings {args.embeddings} does not exist.')
    if not 2 <= args.k_min <= args.k_max:
        parser.error('need 2 <= --k_min <= --k_max.')
    if not 0 < args.target_fpr < 1:
        parser.error('--target_fpr must be between 0 and 1.')
    if args.ae_epochs < 1 or args.ae_bottleneck < 1 or args.ae_hidden < 1 or args.n_init < 1:
        parser.error('--ae_epochs, --ae_bottleneck, --ae_hidden and --n_init must be >= 1.')
    try:
        X, labels, rest = load_embeddings(args.embeddings, args.label_column or None, args.feature_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    if args.outlier_label and labels is None:
        parser.error('--outlier_label needs a label column.')
    roles = None
    if args.role_column:
        if args.role_column not in rest.columns:
            parser.error(f'--role_column {args.role_column} is not a column of the table.')
        roles = rest[args.role_column].astype(str).str.lower().replace({'train': 'fit', 'validation': 'val'}).to_numpy()
    settings = AnalysisSettings(ks=tuple(range(args.k_min, args.k_max + 1)), n_init=args.n_init, linkage=args.linkage,
                                pca_components=args.pca_components, target_fpr=args.target_fpr,
                                outlier_labels=tuple(args.outlier_label), seed=args.seed, plots=not args.no_plots,
                                autoencoder=AutoencoderSettings(hidden=args.ae_hidden, bottleneck=args.ae_bottleneck,
                                                                epochs=args.ae_epochs, seed=args.seed))
    summary = analyse(X, labels, args.out, settings, roles=roles, extra=rest)
    summary['embeddings'] = str(args.embeddings)
    summary['arguments'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    km = summary['kmeans']
    print(f"{summary['rows']} clips, {summary['dimensions']} dims. PCA: first two components explain "
          f"{summary['pca']['first_two']:.1%}. K-Means suggests k={km['suggested_k_silhouette']} (silhouette), "
          f"elbow at k={km['elbow_k']}." + (f" ARI vs labels {km['agreement']['ari']:.3f}." if 'agreement' in km else ''))
    a = summary['anomaly']
    print(f"Anomaly threshold {a['threshold']:.4g} (target FPR {a['target_fpr']}): {a['flagged']} clip(s) flagged"
          + (f", AUC {a['auc']:.3f}" if a.get('auc') is not None else '') + f". Outputs in {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
