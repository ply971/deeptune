"""CLI flags of the evaluation suite (Phase 5) and the df-analyze export. Torch-free, like handlers/video_args.py."""
from pathlib import Path

VALIDATION_SCHEMES = ('holdout', 'kfold', 'bootstrap632')
DFANALYZE_EXTRAS = ('tta', 'motion', 'ids')
DEFAULT_REPORT_BOOTSTRAPS = 2000

EVALUATION_ARG_NAMES = ('validation', 'k', 'n_boot', 'keep_fold_models', 'report_boot')
DFANALYZE_ARG_NAMES = ('export_dfanalyze', 'dfanalyze_extras', 'dfanalyze_clips', 'dfanalyze_dims', 'dfanalyze_rows')
DFANALYZE_EMBED_ARG_NAMES = ('dfanalyze_train_df', 'dfanalyze_val_df')


def _at_least(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise ValueError(f'must be at least {minimum}')
        return value
    parse.__name__ = f'integer of at least {minimum}'
    return parse


def add_evaluation_arguments(parser) -> None:
    parser.add_argument('--validation', choices=VALIDATION_SCHEMES, default='holdout',
                        help="""Video only. How to estimate performance besides the single test score (default holdout: only that).
kfold: k fresh models on the train + validation rows, stratified, and grouped when a group column exists.
bootstrap632: Efron's .632+ on --n_boot fresh models, drawing whole groups when a group column exists.
The test split is never given to these. It is scored once, at the end, by the usual final model.
Results go in validation_<scheme>/ (cv_summary.json, fold_metrics.csv, assignments.csv, oof_predictions.csv).
Training time is multiplied by the number of models.""")
    parser.add_argument('--k', type=_at_least(2), default=5, help='Folds for --validation kfold (default 5).')
    parser.add_argument('--n_boot', type=_at_least(1), default=50,
                        help='Resamples for --validation bootstrap632, each a full training, plus one on all the rows (default 50).')
    parser.add_argument('--keep_fold_models', action='store_true',
                        help='Keep the weights and data subsets of every fold. By default only scores and metrics stay.')
    parser.add_argument('--report_boot', type=_at_least(0), default=DEFAULT_REPORT_BOOTSTRAPS,
                        help=f"""Video only. Bootstrap resamples for the intervals in evaluation_report.json (default {DEFAULT_REPORT_BOOTSTRAPS}).
They resample whole groups when a group column exists. 0 writes no report and skips the extra pass over validation.""")


def add_dfanalyze_arguments(parser, embed_run: bool = False) -> None:
    parser.add_argument('--export_dfanalyze', action='store_true',
                        help="""Video only. Write clip embeddings and labels as CSV files for df-analyze, in dfanalyze_export_*/.
dfanalyze_export.json holds the command to run. The CSV holds only features and the target unless --dfanalyze_extras ids is given.""")
    parser.add_argument('--dfanalyze_extras', nargs='*', choices=DFANALYZE_EXTRAS, default=[], metavar='EXTRA',
                        help="""Optional additions to the export (default none):
tta: mean and standard deviation of the embedding over --dfanalyze_clips clips.
motion: motion_mean, motion_std, motion_max, motion_p90 from a frame-difference curve.
ids: group, split and video_id columns. These are not features: the printed command passes them to df-analyze's --grouper and --drops.""")
    parser.add_argument('--dfanalyze_clips', type=_at_least(2), default=5,
                        help='Clips per video for --dfanalyze_extras tta (default 5). Every split is sampled deterministically, as at test time.')
    parser.add_argument('--dfanalyze_dims', type=_at_least(1), default=None, metavar='N',
                        help="""Reduce the embedding to N PCA components (default: keep every dimension; df-analyze advises under 100 features).
PCA is fitted on the training rows only and then applied to every split. With tta the standard deviations get their own N.""")
    parser.add_argument('--dfanalyze_rows', choices=('test', 'all'), default='test',
                        help="""Rows of dfanalyze.csv, the file the command reads (default test: rows the trained model has not seen).
all: every split. df-analyze's own holdout then includes rows the model trained on, so its scores read too high.
dfanalyze_train.csv, dfanalyze_val.csv and dfanalyze_test.csv are written either way.""")
    if embed_run:
        parser.add_argument('--dfanalyze_train_df', type=Path, default=None,
                            help='The training rows of this model, as a Parquet or CSV file. Exported too, and required with --dfanalyze_dims, as PCA is fitted on them.')
        parser.add_argument('--dfanalyze_val_df', type=Path, default=None, help='The validation rows of this model, exported too. Optional.')
