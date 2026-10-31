"""CLI flags of abstention ("not sure" with a target error rate). Torch-free, like handlers/evaluation_args.py."""
from evaluators.abstention import DEFAULT_CALIB_FRACTION, DEFAULT_DELTA, GUARANTEES, SCORES

ABSTENTION_ARG_NAMES = ('target_error', 'abstain_delta', 'abstain_guarantee', 'abstain_score', 'calib_fraction')


def _open_unit_interval(text: str) -> float:
    value = float(text)
    if not 0 < value < 1:
        raise ValueError('must be between 0 and 1')
    return value


def add_abstention_arguments(parser) -> None:
    parser.add_argument('--target_error', type=_open_unit_interval, default=None, metavar='RATE',
                        help="""Video classification only. Off by default. Fit a threshold so that the model says "not sure" rather than exceed this error rate
among the clips it answers, e.g. 0.05. It promises no accuracy: a model that cannot reach the target answers little or nothing, and the output says so.
Turning this on takes --calib_fraction of the data out of training, whole groups at a time. Validation and test rows do not change.
Fitted on the separate calibration split, saved as abstention.json next to the weights, and applied by infer.py.""")
    parser.add_argument('--abstain_delta', type=_open_unit_interval, default=DEFAULT_DELTA, metavar='DELTA',
                        help=f"""With --target_error and --abstain_guarantee selective_risk: the chance that the calibration draw is unlucky and the guarantee fails (default {DEFAULT_DELTA}).""")
    parser.add_argument('--abstain_guarantee', choices=GUARANTEES, default='selective_risk',
                        help="""With --target_error. What the target bounds.
selective_risk (default): error among answered clips, with probability 1 - delta over the calibration draw, group-weighted, for i.i.d. groups.
silent_error: the chance that a new group has any answered wrong clip, marginally, for exchangeable groups. It does not bound error among answered clips.""")
    parser.add_argument('--abstain_score', choices=SCORES, default='max_prob',
                        help="""With --target_error. What ranks clips (default max_prob, the softmax confidence with no temperature). expected_error uses error_model.json. Validity needs neither to be calibrated.""")
    parser.add_argument('--calib_fraction', type=_open_unit_interval, default=DEFAULT_CALIB_FRACTION, metavar='FRACTION',
                        help=f"""With --target_error. Share of the whole dataset to hold out for calibration, taken from the training split as whole groups (default {DEFAULT_CALIB_FRACTION}).
The effective sample size is the number of groups, so more groups tighten the result; more clips per group mostly do not.""")
