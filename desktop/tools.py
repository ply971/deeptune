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


# select_model.py refuses a strategy's own flags under another strategy (select_model.STRATEGY_FLAGS, kept equal by a
# test), and --include under any strategy but smart (select_model.options_from_args, with its own message).
SELECT_STRATEGY_FLAGS = {('smart',): ('screen', 'screen_views', 'screen_epochs', 'top_families', 'top_backbones', 'tune_epochs',
                                      'recheck_models', 'recheck_full_product', 'final_top'),
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
    if cfg.values.get('include') and strategy != 'smart':
        issues.append("Keep past the cuts (--include) keeps values past the smart search's screening cuts; the other strategies "
                      'have no cuts. Use Fix or Exclude.')
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
def _guarded(title: str, path: Path, build, sections: list) -> None:
    """Run one reader block; a file with unexpected content becomes a "could not read" section, never an exception."""
    try:
        built = build()
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, OSError) as exc:
        sections.append(Section(f'Could not read {title}', notes=[f'{Path(path).name}: {type(exc).__name__}: {exc}'], path=path))
        return
    if built is not None:
        sections.extend(built if isinstance(built, list) else [built])


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


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


def _metric_rows(metrics, intervals=None) -> list:
    rows, intervals = [], _dict(intervals)
    for name, value in _dict(metrics).items():
        if isinstance(value, (dict, list)):
            continue
        rows.append((name, number(value) + (f'  {interval_text(intervals.get(name))}' if intervals else '')))
    return rows


def run_reports(run_dir: Path) -> list[Section]:
    """What a deeptune.py run wrote beyond full_metrics.json: the evaluation report with intervals, the k-fold or bootstrap
    validation summary, the abstention report, and the df-analyze export."""
    run_dir = Path(run_dir)
    sections, warnings = [], []
    report_path = _latest(run_dir, 'eval*/evaluation_report.json')
    if report_path is not None and (report := read_json(report_path, warnings)) is not None:
        def test_report():
            section = Section(f"Test report ({report.get('n', '?')} clips, {report.get('split', 'test')})",
                              _metric_rows(report.get('metrics'), report.get('intervals')), path=report_path)
            section.notes += [str(w) for w in _list(report.get('warnings'))]
            return section
        _guarded('the test report', report_path, test_report, sections)
    for summary_path in sorted(run_dir.glob('validation_*/cv_summary.json')):
        summary = read_json(summary_path, warnings)
        if summary is None:
            continue

        def validation(summary=summary, summary_path=summary_path):
            scheme = summary.get('scheme', summary_path.parent.name)
            rows = [('rows used', number(summary.get('n_rows'))), ('grouped', str(summary.get('grouped')))]
            if summary.get('k'):
                rows.append(('folds', number(summary['k'])))
            oof = _dict(summary.get('out_of_fold'))
            rows += [(f'out-of-fold {name}', value) for name, value in _metric_rows(oof.get('metrics'), oof.get('intervals'))]
            for name, stats in _dict(summary.get('across_folds')).items():
                if isinstance(stats, dict) and stats.get('mean') is not None:
                    rows.append((f'{name} across folds', f"mean {number(stats.get('mean'))}, sd {number(stats.get('std'))}"))
            b632 = _dict(summary.get('bootstrap632'))
            for name, value in _dict(b632.get('estimates') or b632.get('metrics')).items():
                if not isinstance(value, (dict, list)):
                    rows.append((f'.632+ {name}', number(value)))
            return Section(f'Validation: {scheme}', rows, [str(c) for c in _list(summary.get('caveats'))], path=summary_path)
        _guarded('the validation summary', summary_path, validation, sections)
    abstention_path = _latest(run_dir, 'eval*/abstention_report.json')
    if abstention_path is not None and (report := read_json(abstention_path, warnings)) is not None:
        def abstention():
            chosen = _dict(report.get('at_fitted_threshold'))
            rows = [('target error', number(report.get('target_error'))), ('guarantee', str(report.get('guarantee_kind'))),
                    ('status', str(report.get('status'))), ('guarantee holds', str(report.get('guarantee_holds'))),
                    ('coverage (answered share)', number(chosen.get('coverage'))),
                    ('error among answered clips', number(chosen.get('error')) + f"  {interval_text(chosen.get('error_interval'))}"),
                    ('"not sure" clips', number(chosen.get('not_sure_clips')))]
            notes = [str(report['headline'])] if report.get('headline') else []
            return Section('Not sure (abstention) on the test split', rows,
                           notes + [str(w) for w in _list(report.get('warnings'))], path=abstention_path)
        _guarded('the abstention report', abstention_path, abstention, sections)
    export_path = _latest(run_dir, 'dfanalyze_export*/dfanalyze_export.json')
    if export_path is not None and (export := read_json(export_path, warnings)) is not None:
        def dfanalyze():
            rows = [(key, str(value)) for key, value in export.items() if isinstance(value, (str, int, float)) and key != 'command']
            notes = [f"Command: {export['command']}"] if isinstance(export.get('command'), str) else []
            return Section('df-analyze export', rows[:12], notes, path=export_path)
        _guarded('the df-analyze export', export_path, dfanalyze, sections)
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def inference_reports(out_dir: Path) -> list[Section]:
    """The abstention summary of an infer.py run, and the ensemble record when it was an ensemble."""
    out_dir, sections, warnings = Path(out_dir), [], []
    overrides = read_json(out_dir / 'overrides.json', warnings) if (out_dir / 'overrides.json').exists() else None
    if overrides:
        def overrides_section():
            rows = [(f'test-time: {k}', str(v)) for k, v in _dict(overrides.get('test_time_overrides')).items()]
            rows += [(f'model: {k}', str(v)) for k, v in _dict(overrides.get('model_overrides')).items()]
            notes = [str(overrides['warning'])] if overrides.get('warning') else []
            return Section('Overrides of the recorded settings', rows, notes, path=out_dir / 'overrides.json')
        _guarded('overrides.json', out_dir / 'overrides.json', overrides_section, sections)
    summary = read_json(out_dir / 'abstention_summary.json', warnings) if (out_dir / 'abstention_summary.json').exists() else None
    if summary:
        _guarded('the abstention summary', out_dir / 'abstention_summary.json', lambda: Section(
            'Not sure (abstention)', [(key, number(value) if isinstance(value, (int, float)) else str(value))
                                      for key, value in summary.items() if not isinstance(value, (dict, list))],
            path=out_dir / 'abstention_summary.json'), sections)
    sections += ensemble_sections(out_dir, warnings)
    if warnings:
        sections.append(Section('Could not read', notes=warnings))
    return sections


def ensemble_sections(out_dir: Path, warnings: Optional[list] = None) -> list[Section]:
    out_dir, warnings = Path(out_dir), warnings if warnings is not None else []
    sections = []
    record_path = out_dir / 'ensemble.json'
    if record_path.exists() and (record := read_json(record_path, warnings)) is not None:
        def ensemble():
            rows = [(key, str(value)) for key, value in record.items() if isinstance(value, (str, int, float, bool))]
            if isinstance(record.get('members'), list):
                rows.append(('members', str(len(record['members']))))
            return Section('Ensemble', rows, [str(n) for n in _list(record.get('notes'))], path=record_path)
        _guarded('ensemble.json', record_path, ensemble, sections)
    metrics_path = out_dir / 'inference_metrics.json'
    if metrics_path.exists() and (metrics := read_json(metrics_path, warnings)) is not None:
        _guarded('the metrics', metrics_path, lambda: Section('Metrics on the labelled data', _metric_rows(metrics), path=metrics_path),
                 sections)
    predictions = out_dir / 'predictions.csv'
    if predictions.exists():
        try:
            sections.append(Section('Predictions (first 50 rows)', table=pd.read_csv(predictions, nrows=50), path=predictions))
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            warnings.append(f'predictions.csv: {exc}')
    return sections


HONEST_NOTE = ('No stage of the search read the sealed test set; it scored the final winner once. The cross-validation scores '
               'below picked the winner, so they flatter it.')


def sealed_test_section(selection: dict, path: Path) -> Section:
    """The sealed-test block, worded from what selection.json records: the honest number only when this sealed set has
    scored exactly one winner and that score exists."""
    test = _dict(selection.get('test'))
    metrics = _dict(test.get('metrics'))
    uses = selection.get('sealed_test_uses')
    final = _dict(selection.get('final'))
    rows = _metric_rows(metrics, test.get('intervals'))
    if not metrics:
        reason = final.get('reason') or selection.get('reason') or 'the search has not finished.'
        notes = [f'Not scored: {reason}']
        if isinstance(uses, int) and uses >= 1:
            notes.append(f'This sealed test set has already scored {uses} winner(s) of this folder.')
        return Section('Sealed test score: not scored', rows, notes, path=path)
    if uses == 1:
        return Section('Sealed test score (the honest number)', rows + [('times this sealed test set has scored a winner', '1')],
                       [HONEST_NOTE], path=path)
    if isinstance(uses, int) and uses > 1:
        return Section(f'Sealed test score (sealed set reused {uses} times: not an unbiased estimate)',
                       rows + [('times this sealed test set has scored a winner', str(uses))],
                       [f'This sealed test set was reused: it has scored {uses} different winners of this folder '
                        '(--rescore_sealed_test). A test set read more than once is no longer an unbiased estimate of how the '
                        'model does on new data.'], path=path)
    return Section('Sealed test score (use count not recorded)', rows,
                   ['selection.json does not say how many winners this sealed test set has scored, so this page cannot tell '
                    'whether the number is untouched.'], path=path)


def selection_sections(out_dir: Path) -> list[Section]:
    """select_model.py's results: the sealed-test score (the honest number when the sealed set was used once), the winner
    and the leaderboard."""
    out_dir, sections, warnings = Path(out_dir), [], []
    selection_path = out_dir / 'selection.json'
    selection = read_json(selection_path, warnings) if selection_path.exists() else None
    if selection is not None:
        _guarded('the sealed test score', selection_path, lambda: sealed_test_section(selection, selection_path), sections)

        def winner_section():
            winner = _dict(selection.get('winner'))
            rows = [('winner', str(winner.get('label', 'none')))]
            rows += [(key, number(value)) for key, value in winner.items() if key.startswith('cv_') and not isinstance(value, (dict, list))]
            if isinstance(selection.get('runner_up'), dict):
                rows.append(('runner-up', str(selection['runner_up'].get('label'))))
            notes = [str(selection.get('reason', ''))] + [str(n) for n in _list(selection.get('notes'))]
            return Section('Selection', rows, notes, path=selection_path)
        _guarded('the selection', selection_path, winner_section, sections)
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
        _guarded('the probe metrics', metrics_path, lambda: Section('Probe test metrics', _metric_rows(metrics), path=metrics_path),
                 sections)
    audit_path = out_dir / 'split_audit.json'
    if audit_path.exists() and (audit := read_json(audit_path, warnings)) is not None:
        _guarded('the split audit', audit_path, lambda: Section(
            'Split audit', [(k, str(v)) for k, v in audit.items() if not isinstance(v, (dict, list))], path=audit_path), sections)
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
