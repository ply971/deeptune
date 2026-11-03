"""The GUI side of the other entry points (G1): select_model.py, ensemble.py, extract_features.py and
train_feature_head.py, plus readers for what runs leave on disk (selection leaderboard, validation report, abstention,
ensemble and probe results).

Pure Python (json, csv via pandas, pathlib): no torch and no Dear PyGui, so tests drive it directly. The widgets are
in desktop/tool_pages.py; the processes start through desktop/runner.start_tool_run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from desktop import cli_options as co


@dataclass(frozen=True)
class Tool:
    name: str
    script: str
    specs: tuple
    title: str


TOOLS = {tool.name: tool for tool in (
    Tool('select_model', 'select_model.py', co.SELECT_OPTIONS, 'Model selection'),
    Tool('ensemble', 'ensemble.py', co.ENSEMBLE_OPTIONS, 'Ensemble'),
    Tool('extract_features', 'extract_features.py', co.EXTRACT_OPTIONS, 'Feature extraction'),
    Tool('train_feature_head', 'train_feature_head.py', co.PROBE_OPTIONS, 'Feature probe'),
)}


@dataclass
class ToolConfig:
    tool: str
    values: dict = field(default_factory=dict)

    @property
    def spec(self) -> Tool:
        return TOOLS[self.tool]

    def get(self, key):
        return self.values.get(key, co.by_key(self.spec.specs)[key].empty)


def build_tool_argv(cfg: ToolConfig) -> list[str]:
    """The entry point's arguments: every value that differs from the CLI default, required ones always."""
    return co.options_argv(cfg.spec.specs, cfg.values)


def output_folder(cfg: ToolConfig) -> Optional[Path]:
    key = 'out'
    value = cfg.get(key)
    return Path(value).expanduser() if value else None


def validate_tool(cfg: ToolConfig) -> list[str]:
    values, issues = co.coerce_values(cfg.spec.specs, cfg.values)
    if issues:
        return issues
    for opt in cfg.spec.specs:
        if opt.required and not values.get(opt.key):
            issues.append(f'{opt.label} is required.')
    check = {'select_model': _select_issues, 'ensemble': _ensemble_issues, 'extract_features': _extract_issues,
             'train_feature_head': _probe_issues}[cfg.tool]
    return issues + check(ToolConfig(cfg.tool, values))


def _exists(label: str, path) -> list[str]:
    return [] if not path or Path(path).expanduser().exists() else [f'{label} not found: {path}']


# select_model.py refuses a strategy's own flags under another strategy (select_model.STRATEGY_FLAGS).
SELECT_STRATEGY_FLAGS = {('smart',): ('screen', 'screen_views', 'screen_epochs', 'top_families', 'top_backbones', 'tune_epochs',
                                      'recheck_models', 'recheck_full_product', 'final_top', 'include'),
                         ('random', 'bayes'): ('n_trials',), ('bayes',): ('bayes_startup',)}


def _select_issues(cfg: ToolConfig) -> list[str]:
    issues = _exists('Dataset', cfg.get('df')) + _exists('Grid file', cfg.get('grid'))
    if cfg.get('k') < 2:
        issues.append('Folds (k) must be at least 2.')
    if not 0 < cfg.get('test_fraction') <= 0.5:
        issues.append('The sealed test share must be in (0, 0.5].')
    if set(cfg.get('augmentations')) & {'aug_flip', 'aug_mix_flip'} and not cfg.get('allow_flip'):
        issues.append('Flip augmentations turn left motion into right motion: tick "My labels do not depend on direction" '
                      'only if that is true.')
    if not cfg.get('backbones') and not cfg.get('grid'):
        issues.append('Name at least one backbone, or choose a grid file.')
    strategy = cfg.get('search')
    for owners, keys in SELECT_STRATEGY_FLAGS.items():
        given = [co.by_key(co.SELECT_OPTIONS)[key].label for key in keys if cfg.values.get(key) not in (None, [], ())]
        if given and strategy not in owners:
            issues.append(f'{", ".join(given)}: only for the {" or ".join(owners)} strategy; clear them or change the strategy.')
    metric = cfg.get('metric')
    if metric is not None and metric not in (co.SELECTION_METRICS[:5] if cfg.get('mode') == 'cls' else co.SELECTION_METRICS[5:]):
        issues.append(f'The metric {metric} does not fit the task {cfg.get("mode")}.')
    for key in ('learning_rates',):
        for word in cfg.get(key):
            try:
                float(word)
            except ValueError:
                issues.append(f'Learning rates must be numbers, not {word!r}.')
    gpu = str(cfg.get('gpu_memory_gb')).strip().lower()
    if gpu not in ('auto', 'none'):
        try:
            float(gpu)
        except ValueError:
            issues.append('GPU memory must be auto, none or a number of GB.')
    return issues


def _ensemble_issues(cfg: ToolConfig) -> list[str]:
    runs, predictions = cfg.get('runs'), cfg.get('predictions')
    if bool(runs) == bool(predictions):
        return ['Give either trained runs or predictions files, not both.' if runs else 'Give trained runs or predictions files.']
    issues = []
    members = runs or predictions
    if len(members) < 2:
        issues.append('An ensemble needs at least two members, one per line.')
    for member in members:
        issues += _exists('Member', member)
    if runs and not cfg.get('data'):
        issues.append('Trained runs need data to predict on.')
    issues += _exists('Data', cfg.get('data')) + _exists('Label mapping', cfg.get('label_mapping'))
    if cfg.get('batch_size') < 1:
        issues.append('Batch size must be at least 1.')
    return issues


def _extract_issues(cfg: ToolConfig) -> list[str]:
    issues = _exists('Videos', cfg.get('data'))
    if not cfg.get('streams'):
        issues.append('Choose at least one stream.')
    for key in ('snippet_frames', 'batch_size'):
        if cfg.get(key) < 1:
            issues.append(f'{co.by_key(co.EXTRACT_OPTIONS)[key].label} must be at least 1.')
    if cfg.get('motion_gain') <= 0:
        issues.append('Motion gain must be positive.')
    return issues


def _probe_issues(cfg: ToolConfig) -> list[str]:
    issues = (_exists('Feature folder', cfg.get('features')) + _exists('Training rows', cfg.get('train_df'))
              + _exists('Validation rows', cfg.get('val_df')) + _exists('Test rows', cfg.get('test_df')))
    if cfg.get('mode') == 'cls' and cfg.get('num_classes') < 2:
        issues.append('Number of classes must be at least 2 for classification.')
    if not 1 <= cfg.get('probe_depth') <= 4:
        issues.append('Probe depth must be 1 to 4.')
    return issues


# ---------------------------------------------------------------------------------------------------------------------
# Reading results back (best effort: a missing or broken file becomes a warning, never an exception)
# ---------------------------------------------------------------------------------------------------------------------
def read_json(path: Path, warnings: Optional[list] = None) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        if warnings is not None:
            warnings.append(f'{Path(path).name}: {exc}')
        return None
    return value if isinstance(value, dict) else None


def _latest(root: Path, pattern: str) -> Optional[Path]:
    found = sorted(Path(root).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def number(value) -> str:
    if isinstance(value, bool) or value is None:
        return 'Unavailable' if value is None else str(value)
    if isinstance(value, (int, float)):
        return f'{value:.4g}' if math.isfinite(value) else 'Unavailable'
    return str(value)


def interval_text(interval) -> str:
    if not isinstance(interval, dict) or interval.get('low') is None:
        return ''
    level = interval.get('level')
    level_text = f'{level:.0%} ' if isinstance(level, (int, float)) else ''
    return f"{level_text}[{number(interval['low'])}, {number(interval['high'])}]"


@dataclass
class Section:
    """One block of a results view: a title, (label, value) rows, notes and an optional table."""
    title: str
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    table: Optional[pd.DataFrame] = None
    path: Optional[Path] = None


def _metric_rows(metrics: dict, intervals: Optional[dict] = None) -> list:
    rows = []
    for name, value in (metrics or {}).items():
        if isinstance(value, dict):
            continue
        rows.append((name, number(value) + (f'  {interval_text((intervals or {}).get(name))}' if intervals else '')))
    return rows


def run_reports(run_dir: Path) -> list[Section]:
    """What a deeptune.py run wrote beyond full_metrics.json: the evaluation report with intervals, the k-fold or bootstrap
    validation summary, the abstention report, and the df-analyze export."""
    run_dir = Path(run_dir)
    sections, warnings = [], []
    report_path = _latest(run_dir, 'eval*/evaluation_report.json')
    if report_path is not None and (report := read_json(report_path, warnings)) is not None:
        section = Section(f"Test report ({report.get('n', '?')} clips, {report.get('split', 'test')})",
                          _metric_rows(report.get('metrics', {}), report.get('intervals', {})), path=report_path)
        section.notes += [str(w) for w in report.get('warnings', [])]
        sections.append(section)
    for summary_path in sorted(run_dir.glob('validation_*/cv_summary.json')):
        summary = read_json(summary_path, warnings)
        if summary is None:
            continue
        scheme = summary.get('scheme', summary_path.parent.name)
        rows = [('rows used', number(summary.get('n_rows'))), ('grouped', str(summary.get('grouped')))]
        if summary.get('k'):
            rows.append(('folds', number(summary['k'])))
        oof = summary.get('out_of_fold') or {}
        rows += [(f'out-of-fold {name}', value) for name, value in _metric_rows(oof.get('metrics', {}), oof.get('intervals', {}))]
        for name, stats in (summary.get('across_folds') or {}).items():
            if isinstance(stats, dict) and stats.get('mean') is not None:
                rows.append((f'{name} across folds', f"mean {number(stats.get('mean'))}, sd {number(stats.get('std'))}"))
        b632 = summary.get('bootstrap632') or {}
        for name, value in (b632.get('estimates') or b632.get('metrics') or {}).items():
            if not isinstance(value, dict):
                rows.append((f'.632+ {name}', number(value)))
        section = Section(f'Validation: {scheme}', rows, list(summary.get('caveats', [])), path=summary_path)
        sections.append(section)
    abstention_path = _latest(run_dir, 'eval*/abstention_report.json')
    if abstention_path is not None and (report := read_json(abstention_path, warnings)) is not None:
        chosen = report.get('at_fitted_threshold') or {}
        rows = [('target error', number(report.get('target_error'))), ('guarantee', str(report.get('guarantee_kind'))),
                ('status', str(report.get('status'))), ('guarantee holds', str(report.get('guarantee_holds'))),
                ('coverage (answered share)', number(chosen.get('coverage'))),
                ('error among answered clips', number(chosen.get('error')) + f"  {interval_text(chosen.get('error_interval'))}"),
                ('"not sure" clips', number(chosen.get('not_sure_clips')))]
        notes = [report['headline']] if report.get('headline') else []
        sections.append(Section('Not sure (abstention) on the test split', rows, notes + list(report.get('warnings', [])),
                                path=abstention_path))
    export_path = _latest(run_dir, 'dfanalyze_export*/dfanalyze_export.json')
    if export_path is not None and (export := read_json(export_path, warnings)) is not None:
        rows = [(key, str(value)) for key, value in export.items() if isinstance(value, (str, int, float)) and key != 'command']
        notes = [f"Command: {export['command']}"] if isinstance(export.get('command'), str) else []
        sections.append(Section('df-analyze export', rows[:12], notes, path=export_path))
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def inference_reports(out_dir: Path) -> list[Section]:
    """The abstention summary of an infer.py run, and the ensemble record when it was an ensemble."""
    out_dir, sections, warnings = Path(out_dir), [], []
    summary = read_json(out_dir / 'abstention_summary.json', warnings) if (out_dir / 'abstention_summary.json').exists() else None
    if summary:
        rows = [(key, number(value) if isinstance(value, (int, float)) else str(value)) for key, value in summary.items()
                if not isinstance(value, (dict, list))]
        sections.append(Section('Not sure (abstention)', rows, path=out_dir / 'abstention_summary.json'))
    sections += ensemble_sections(out_dir, warnings)
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def ensemble_sections(out_dir: Path, warnings: Optional[list] = None) -> list[Section]:
    out_dir, warnings = Path(out_dir), warnings if warnings is not None else []
    sections = []
    record_path = out_dir / 'ensemble.json'
    if record_path.exists() and (record := read_json(record_path, warnings)) is not None:
        rows = [(key, str(value)) for key, value in record.items() if isinstance(value, (str, int, float, bool))]
        members = record.get('members')
        if isinstance(members, list):
            rows.append(('members', str(len(members))))
        notes = [str(n) for n in record.get('notes', [])] if isinstance(record.get('notes'), list) else []
        sections.append(Section('Ensemble', rows, notes, path=record_path))
    metrics_path = out_dir / 'inference_metrics.json'
    if metrics_path.exists() and (metrics := read_json(metrics_path, warnings)) is not None:
        sections.append(Section('Metrics on the labelled data', _metric_rows(metrics), path=metrics_path))
    predictions = out_dir / 'predictions.csv'
    if predictions.exists():
        try:
            sections.append(Section('Predictions (first 50 rows)', table=pd.read_csv(predictions, nrows=50), path=predictions))
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            warnings.append(f'predictions.csv: {exc}')
    return sections


def selection_sections(out_dir: Path) -> list[Section]:
    """select_model.py's results: the winner, the sealed-test score (the honest number) and the leaderboard."""
    out_dir, sections, warnings = Path(out_dir), [], []
    selection = read_json(out_dir / 'selection.json', warnings) if (out_dir / 'selection.json').exists() else None
    if selection is not None:
        test = selection.get('test') or {}
        metrics = test.get('metrics') or {}
        honest = Section('Sealed test score (the honest number)', _metric_rows(metrics, test.get('intervals', {})),
                         ['No stage of the search read the sealed test set; it scored the final winner once. The cross-validation '
                          'scores below picked the winner, so they flatter it.'], path=out_dir / 'selection.json')
        if not metrics:
            honest.notes.insert(0, 'No sealed-test score yet: ' + str((selection.get('final') or {}).get('reason')
                                                                       or selection.get('reason') or 'the search has not finished.'))
        uses = selection.get('sealed_test_uses')
        if uses is not None:
            honest.rows.append(('times this sealed test set has scored a winner', number(uses)))
        sections.append(honest)
        winner = selection.get('winner') or {}
        rows = [('winner', str(winner.get('label', 'none')))]
        rows += [(key, number(value)) for key, value in winner.items() if key.startswith('cv_')]
        if selection.get('runner_up'):
            rows.append(('runner-up', str(selection['runner_up'].get('label'))))
        sections.append(Section('Selection', rows, [str(selection.get('reason', ''))] + [str(n) for n in selection.get('notes', [])],
                                path=out_dir / 'selection.json'))
    leaderboard = out_dir / 'leaderboard.csv'
    if leaderboard.exists():
        try:
            sections.append(Section('Leaderboard (cross-validation on the development rows)', table=pd.read_csv(leaderboard),
                                    path=leaderboard))
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            warnings.append(f'leaderboard.csv: {exc}')
    if (out_dir / 'ledger.jsonl').exists() and selection is None:
        sections.append(Section('Search in progress or stopped', notes=[
            'This folder has a ledger of finished trainings. Start the search again with the same folder to resume; '
            '"Resume with a changed grid" and "Retry failed candidates" are under Limits, resume and checks.']))
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def is_resumable(out_dir) -> bool:
    return bool(out_dir) and (Path(out_dir).expanduser() / 'ledger.jsonl').exists()


def probe_sections(out_dir: Path) -> list[Section]:
    out_dir, sections, warnings = Path(out_dir), [], []
    metrics_path = out_dir / 'metrics.json'
    if metrics_path.exists() and (metrics := read_json(metrics_path, warnings)) is not None:
        sections.append(Section('Probe test metrics', _metric_rows(metrics), path=metrics_path))
    audit_path = out_dir / 'split_audit.json'
    if audit_path.exists() and (audit := read_json(audit_path, warnings)) is not None:
        sections.append(Section('Split audit', [(k, str(v)) for k, v in audit.items() if not isinstance(v, (dict, list))],
                                path=audit_path))
    predictions = out_dir / 'predictions.csv'
    if predictions.exists():
        try:
            sections.append(Section('Predictions (first 50 rows)', table=pd.read_csv(predictions, nrows=50), path=predictions))
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            warnings.append(f'predictions.csv: {exc}')
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def feature_sections(out_dir: Path) -> list[Section]:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    files = [p for p in out_dir.rglob('*') if p.is_file()]
    rows = [('files', str(len(files)))]
    record = out_dir / 'cli_arguments.json'
    notes = []
    if record.exists() and (args := read_json(record, notes)) is not None:
        rows += [(key, str(args.get(key))) for key in ('model_version', 'streams', 'snippet_frames', 'cache_tokens') if key in args]
    return [Section('Feature folder', rows, notes, path=out_dir)]
