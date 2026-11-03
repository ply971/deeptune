"""Smart search's cheap screen: a frozen backbone's cached snippet features and an attentive probe, on the same folds.

For a candidate with a frozen backbone the expensive part of every training is the backbone's forward pass, repeated each
epoch. Here it runs once per backbone: `extract_features.py` writes snippet features of every development clip (B1/R1,
P-102) into `<out>/features/<backbone>-<frames>f`, reused by every fold and every later stage. Then each fold trains an
attentive probe on those features with `train_feature_head.py` (a subprocess, like deeptune.py's runs).

Only dev.parquet is ever given to either script. The folds are deeptune.py --cv_only's own: the rows get the same label
encoding (`prepare_whole_dataset`), the same groups (the group column, else repeated source videos) and the same seed
go to the same `evaluators.video.validation.run_validation`, which writes the same cv_summary.json and out-of-fold files.
The engine's fold check (ranking.fold_signature) confirms it: a probe on other folds would stop the ranking.

A probe's score measures the backbone's frozen features under a probe head, not the candidate's own head and pooling, so it
is a screening score only: the leaderboard lists it without a rank (protocol `probe`) and never against cv scores.
Credit: the attentive probe and the feature cache are DeepTune's B1 / P-102 (src/video/feature_heads.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from handlers.model_selection.grid import DEFAULT, parse_train_mode
from handlers.model_selection.runner import REPO, RunResult, failure_reason, result_from_exit

PROBE_EPOCHS, PROBE_LR, DEFAULT_FRAMES = 10, 1e-3, 8
LOG_FILE = 'probe.log'


def probe_frames(candidate) -> int:
    return DEFAULT_FRAMES if candidate.num_frames == DEFAULT else int(candidate.num_frames)


def probe_problem(candidate, settings) -> Optional[str]:
    """Why this candidate's backbone cannot be screened by a probe on cached features, or None."""
    if parse_train_mode(candidate.train_mode)[0] != 'frozen':
        return f'its backbone trains ({candidate.train_mode}), and a probe on cached features stands only for a frozen one'
    try:
        from extract_features import check_feature_input
        from handlers.video_backbones import hub_backbone
        info = hub_backbone(candidate.backbone)
        pretrained = not (info is not None and info.is_untrained) and not candidate.backbone.endswith('_tiny')
        check_feature_input('rgb', candidate.backbone, pretrained, None, probe_frames(candidate))
    except Exception as exc:  # the extractor's own refusal, in its words
        return f'the feature extractor refuses it ({type(exc).__name__}: {exc})'
    return None


class FeatureProbeRunner:
    """`(job, run_dir, info) -> RunResult`. `info`: dev (path), settings (RunSettings), features_root, mode, num_classes."""

    def __init__(self, python: str = sys.executable, env: Optional[dict] = None, epochs: int = PROBE_EPOCHS):
        self.python, self.epochs = python, int(epochs)
        self.env = dict(os.environ if env is None else env)
        self.env.setdefault('PYTHONIOENCODING', 'utf-8')
        self.env.setdefault('PYTHONUTF8', '1')

    def _call(self, script: str, argv: list, log) -> int:
        log.write(f'# {self.python} {script} {subprocess.list2cmdline([str(a) for a in argv])}\n')
        log.flush()
        process = subprocess.Popen([self.python, '-u', str(REPO / script), *map(str, argv)], stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(REPO), env=self.env)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.kill()
            process.wait()
            raise

    def features(self, candidate, info, log) -> tuple[Path, int]:
        settings = info['settings']
        frames = probe_frames(candidate)
        folder = Path(info['features_root']) / f'{candidate.backbone}-{frames}f'
        from handlers.video_backbones import hub_backbone
        hub = hub_backbone(candidate.backbone)
        untrained = (hub is not None and hub.is_untrained) or candidate.backbone.endswith('_tiny')
        argv = ['--data', str(info['dev']), '--out', str(folder), '--model_version', candidate.backbone, '--streams', 'rgb',
                '--snippet_frames', str(frames), '--feature_weights', 'none' if untrained else 'pretrained',
                '--batch_size', str(max(1, settings.batch_size // 2))]
        if settings.accept_weights_licence:
            argv += ['--accept_weights_licence', *settings.accept_weights_licence]
        return folder, self._call('extract_features.py', argv, log)

    def __call__(self, job, run_dir: Path, info: dict) -> RunResult:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from evaluators.video.run_validation import _column, _holds_paths, _video_ids, read_pool
        from evaluators.video.validation import FoldScores, run_validation
        from handlers.split_dataset import prepare_whole_dataset, resolve_grouper
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        settings, mode, start = info['settings'], info['mode'], time.perf_counter()
        log_path = run_dir / LOG_FILE
        with open(log_path, 'w', encoding='utf-8') as log:
            folder, code = self.features(job.candidate, info, log)
            if code != 0:
                log.flush()
                return result_from_exit(code, time.perf_counter() - start, log_path.read_text(encoding='utf-8', errors='replace')[-20000:])
            grouper = resolve_grouper(pq.read_schema(info['dev']).names, settings.grouper, 'video')
            root = run_dir / 'probe-x'
            pool_path = prepare_whole_dataset(info['dev'], root, False, settings.target, 'video', grouper, mode)
            pool = read_pool(pool_path, None)
            videos = _column(pool, 'videos') if _holds_paths(pool) else None
            groups, group_source = _column(pool, grouper), 'column'
            if groups is None and videos is not None and len(set(videos)) < len(videos):
                groups, group_source = videos, 'source_video'

            def fit_predict(train_rows, val_rows, eval_rows, name):
                fold = root / 'validation_kfold' / name
                fold.mkdir(parents=True, exist_ok=True)
                paths = {role: fold / f'{role}.parquet' for role in ('train', 'val', 'eval')}
                for role, rows in (('train', train_rows), ('val', val_rows), ('eval', eval_rows)):
                    pq.write_table(pool.take(pa.array(np.asarray(rows, dtype=np.int64))), paths[role])
                argv = ['--features', str(folder), '--train_df', str(paths['train']), '--val_df', str(paths['val']),
                        '--test_df', str(paths['eval']), '--out', str(fold / 'probe'), '--head', 'attentive_probe', '--mode', mode,
                        '--num_classes', str(info['num_classes'] if mode == 'cls' else 1), '--batch_size', str(settings.batch_size),
                        '--num_epochs', str(self.epochs), '--learning_rate', str(PROBE_LR), '--fixed-seed']
                if self._call('train_feature_head.py', argv, log) != 0:
                    log.flush()
                    raise RuntimeError(failure_reason(log_path.read_text(encoding='utf-8', errors='replace')[-20000:]))
                import pandas as pd
                table = pd.read_csv(fold / 'probe' / 'predictions.csv')
                if mode == 'cls':
                    columns = sorted((c for c in table.columns if c.startswith('probability_')), key=lambda c: int(c.rsplit('_', 1)[1]))
                    return FoldScores(table[columns].to_numpy())
                return FoldScores(table['prediction'].to_numpy(dtype=float))
            try:
                run_validation('kfold', np.asarray(_column(pool, 'labels')), fit_predict, root / 'validation_kfold', mode=mode,
                               ids=_video_ids(pool), groups=groups, group_source=group_source, sources=videos, k=settings.k,
                               seed=settings.seed, ci_boot=settings.boot)
            except Exception as exc:
                return RunResult('failed', time.perf_counter() - start, f'probe: {type(exc).__name__}: {exc}', 1)
            (root / 'probe_protocol.json').write_text(json.dumps({'features': str(folder), 'head': 'attentive_probe', 'epochs': self.epochs,
                                                                   'learning_rate': PROBE_LR, 'seed': 42, 'folds_seed': settings.seed}, indent=2),
                                                       encoding='utf-8')
        return RunResult('ok', time.perf_counter() - start)
