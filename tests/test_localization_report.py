"""Localisation mAP against the ActivityNet reference as used by CtxPL, and hand-worked examples.

`_ref_*` below are copied from github.com/moayadeldin/CtxPL (MIT), eval/eval_detection.py and eval/utils_eval.py, read on
2026-10-31, with only the imports adapted. They are the oracle: the production code is a separate implementation.
"""
import json

import numpy as np
import pandas as pd
import pytest

from evaluators.video import localization_report as lr

THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 1)


def _ref_interpolated_prec_rec(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def _ref_segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
        + (target_segment[1] - target_segment[0]) - segments_intersection
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def _ref_ap(ground_truth, prediction, tiou_thresholds):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))
    ground_truth_gbvn = ground_truth.groupby("video-id")
    for idx, this_pred in prediction.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception:
            fp[:, idx] = 1
            continue
        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = _ref_segment_iou(this_pred[["t-start", "t-end"]].values.astype(float), this_gt[["t-start", "t-end"]].values.astype(float))
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1
    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = _ref_interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    return ap


def _ref_get_ap(conf, labels):
    sortind = np.argsort(-conf)
    tp = labels[sortind] == 1
    fp = labels[sortind] != 1
    npos = np.sum(labels)
    fp = np.cumsum(fp).astype('float32')
    tp = np.cumsum(tp).astype('float32')
    prec = tp / (fp + tp)
    tmp = (labels[sortind] == 1).astype('float32')
    return np.sum(tmp * prec) / npos


def gt(rows):
    return pd.DataFrame(rows, columns=['video-id', 't-start', 't-end', 'label'])


def pred(rows):
    return pd.DataFrame(rows, columns=['video-id', 't-start', 't-end', 'label', 'score'])


def _ap(truth, prediction, thresholds=THRESHOLDS):
    return lr.class_average_precision(truth, prediction, thresholds)


def test_one_perfect_detection_has_ap_one():
    assert _ap(gt([('v', 1, 3, 0)]), pred([('v', 1, 3, 0, .9)])).tolist() == [1.0] * 9


def test_worked_example_by_hand():
    """Two truths. Ranked: hit, false positive, hit: precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
    Envelope: 1 up to recall 1/2, then 2/3. AP = 1/2 * 1 + 1/2 * 2/3 = 5/6."""
    truth = gt([('a', 0, 10, 0), ('b', 0, 10, 0)])
    prediction = pred([('a', 0, 10, 0, .9), ('c', 0, 10, 0, .8), ('b', 0, 10, 0, .7)])
    assert _ap(truth, prediction, [0.5]) == pytest.approx([5 / 6])


def test_duplicate_detections_score_one_true_positive():
    """Second copy is a false positive: AP stays 1 when the duplicate ranks last, and drops when it ranks before a hit."""
    truth = gt([('a', 0, 10, 0), ('b', 0, 10, 0)])
    assert _ap(truth, pred([('a', 0, 10, 0, .9), ('b', 0, 10, 0, .8), ('a', 0, 10, 0, .7)]), [0.5]) == pytest.approx([1.0])
    assert _ap(truth, pred([('a', 0, 10, 0, .9), ('a', 0, 10, 0, .8), ('b', 0, 10, 0, .7)]), [0.5]) == pytest.approx([5 / 6])


def test_a_lower_scored_prediction_takes_the_next_free_truth():
    truth = gt([('a', 0, 10, 0), ('a', 0, 8, 0)])
    # The top prediction claims (0, 10); the second overlaps it best, finds it taken, and claims (0, 8).
    assert _ap(truth, pred([('a', 0, 10, 0, .9), ('a', 0, 9.9, 0, .8)]), [0.5]) == pytest.approx([1.0])


def test_prediction_in_a_video_without_truth_is_a_false_positive():
    truth = gt([('a', 0, 10, 0)])
    assert _ap(truth, pred([('z', 0, 10, 0, .9), ('a', 0, 10, 0, .8)]), [0.5]) == pytest.approx([0.5])


def test_empty_predictions_score_zero():
    report = lr.detection_map(gt([('a', 0, 1, 0)]), pred([]))
    assert set(report['map_per_threshold'].values()) == {0.0} and report['averages']['avg_0.1_0.5'] == 0.0


def test_touching_segments_do_not_overlap():
    assert lr.segment_iou(np.array([0., 5.]), np.array([[5., 10.]])).tolist() == [0.0]
    assert _ap(gt([('a', 5, 10, 0)]), pred([('a', 0, 5, 0, .9)]), [0.1]).tolist() == [0.0]


def test_iou_exactly_at_threshold_counts():
    truth = gt([('a', 0, 10, 0)])
    assert _ap(truth, pred([('a', 0, 5, 0, .9)]), [0.5]) == pytest.approx([1.0])


def test_zero_length_segments_do_not_divide_by_zero():
    assert lr.segment_iou(np.array([2., 2.]), np.array([[2., 2.]])).tolist() == [0.0]


def test_map_ignores_classes_without_truth_and_averages_ranges():
    truth = gt([('a', 0, 10, 0), ('a', 20, 30, 1)])
    prediction = pred([('a', 0, 10, 0, .9), ('a', 0, 10, 7, .9)])  # class 1 missed, class 7 has no truth
    report = lr.detection_map(truth, prediction)
    assert report['map_per_threshold']['0.5'] == pytest.approx(0.5)
    assert report['ignored_prediction_labels'] == ['7']
    assert report['averages'] == {'avg_0.1_0.5': pytest.approx(.5), 'avg_0.1_0.7': pytest.approx(.5), 'avg_0.1_0.9': pytest.approx(.5)}
    assert report['per_class_ap']['1']['0.5'] == 0.0


def test_averages_use_the_stated_threshold_ranges():
    truth = gt([('a', 0, 10, 0)])
    report = lr.detection_map(truth, pred([('a', 0, 6, 0, .9)]))  # tIoU .6: found up to .6, missed from .7
    assert report['averages']['avg_0.1_0.5'] == pytest.approx(1.0)
    assert report['averages']['avg_0.1_0.7'] == pytest.approx(6 / 7)
    assert report['averages']['avg_0.1_0.9'] == pytest.approx(6 / 9)


def _random_tables(seed):
    rng = np.random.default_rng(seed)
    truths, predictions = [], []
    for v in range(rng.integers(3, 8)):
        for _ in range(rng.integers(1, 5)):
            start = rng.uniform(0, 80)
            truths.append((f'v{v}', start, start + rng.uniform(1, 20), 0))
    for _ in range(rng.integers(1, 40)):
        video = f'v{rng.integers(0, 10)}'  # some videos have no truth
        start = rng.uniform(0, 80)
        predictions.append((video, start, start + rng.uniform(0.5, 20), 0, rng.random()))
    for t in truths[:len(truths) // 2]:  # near copies, and exact duplicates
        predictions.append((t[0], t[1] + rng.normal(0, 1), t[2] + rng.normal(0, 1), 0, rng.random()))
        predictions.append((t[0], t[1], t[2], 0, rng.random()))
    return gt(truths), pred(predictions)


@pytest.mark.parametrize('seed', range(25))
def test_matches_the_reference_on_random_tables(seed):
    truth, prediction = _random_tables(seed)
    assert _ap(truth, prediction) == pytest.approx(_ref_ap(truth, prediction, THRESHOLDS), abs=1e-12)


def test_matches_the_reference_when_scores_tie():
    truth = gt([('a', 0, 10, 0), ('b', 0, 10, 0)])
    prediction = pred([('a', 0, 10, 0, .5), ('x', 0, 10, 0, .5), ('b', 0, 10, 0, .5), ('a', 1, 10, 0, .5)])
    assert _ap(truth, prediction) == pytest.approx(_ref_ap(truth, prediction, THRESHOLDS))


def test_interpolated_ap_matches_the_reference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        recall = np.sort(rng.random(12))
        precision = rng.random(12)
        assert lr.interpolated_ap(precision, recall) == pytest.approx(_ref_interpolated_prec_rec(precision, recall))


def test_classification_map_matches_the_reference_and_a_hand_value():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    scores = np.array([[.9, .1], [.8, .7], [.3, .6], [.1, .2]])
    got = lr.classification_map(scores, labels)
    # class 0: ranked 1,0,1,0 -> (1 + 2/3) / 2 = 5/6. class 1: ranked 1,1,0,0 -> 1.
    assert got['per_class_ap'] == pytest.approx([5 / 6, 1.0])
    assert got['map'] == pytest.approx((5 / 6 + 1) / 2)
    for k in range(2):
        assert got['per_class_ap'][k] == pytest.approx(_ref_get_ap(scores[:, k], labels[:, k]))


def test_classification_map_leaves_out_a_class_without_positives():
    got = lr.classification_map(np.array([[.9, .1], [.2, .8]]), np.array([[1, 0], [1, 0]]))
    assert got['per_class_ap'][1] is None and got['map'] == 1.0


def test_boundary_error_by_hand():
    truth = gt([('a', 10, 20, 0)])
    got = lr.boundary_error(truth, pred([('a', 11, 22, 0, .9), ('b', 0, 1, 0, .8)]))
    assert got == {'tiou': .5, 'n_matched': 1, 'start_seconds': 1.0, 'end_seconds': 2.0, 'mean_seconds': 1.5}
    assert lr.boundary_error(truth, pred([('b', 0, 1, 0, .8)]))['start_seconds'] is None


def test_false_positive_flags_follow_the_prediction_rows():
    truth = gt([('a', 0, 10, 0)])
    prediction = pred([('a', 0, 10, 0, .1), ('a', 0, 10, 0, .9), ('z', 0, 10, 0, .5)])
    assert lr.segment_false_positives(truth, prediction).tolist() == [1, 0, 1]  # the higher score wins the match


def _simulated_uncertainty(seed, stated_from_truth):
    rng = np.random.default_rng(seed)
    truth = rng.random(4000)
    wrong = (rng.random(4000) < truth).astype(float)
    return stated_from_truth(truth, rng), wrong


def test_calibrated_uncertainty_passes():
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t)
    got = lr.validate_uncertainty(stated, wrong)
    assert got['calibrated'] and got['ranks_errors'] and got['auroc_error_detection'] > .7
    assert len(got['table']['bins']) == 10


def test_rank_only_uncertainty_is_reported_as_not_calibrated():
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t ** 4)  # same order, wrong size
    got = lr.validate_uncertainty(stated, wrong)
    assert not got['calibrated'] and got['ranks_errors']
    table = pd.DataFrame({'t-start': np.zeros(len(stated))})
    got = {**got, 'unit': 'segment', 'split': 'validation'}
    assert 'expected_error' not in lr.attach_expected_error(table, stated, got, applied_to='test').columns


def test_unrelated_uncertainty_ranks_nothing():
    stated, wrong = _simulated_uncertainty(1, lambda t, rng: rng.random(len(t)))
    got = lr.validate_uncertainty(stated, wrong)
    assert not got['calibrated'] and abs(got['auroc_error_detection'] - .5) < .05


def test_expected_error_is_attached_when_calibrated():
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t)
    table = pd.DataFrame({'t-start': np.zeros(len(stated))})
    out = lr.attach_expected_error(table, stated, lr.validate_uncertainty(stated, wrong, unit='segment', split='validation'), applied_to='test')
    assert out['expected_error'].tolist() == stated.tolist()


def test_applying_uncertainty_to_the_rows_it_was_checked_on_is_refused():
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t)
    validation = lr.validate_uncertainty(stated, wrong, unit='segment', split='test')
    table = pd.DataFrame({'t-start': np.zeros(len(stated))})
    with pytest.raises(ValueError, match='validated on the test rows'):
        lr.attach_expected_error(table, stated, validation, applied_to='test')
    assert 'expected_error' in lr.attach_expected_error(table, stated, validation, applied_to='new_videos').columns


def _segments_with_known_errors(seed=0, n=3000):
    """Each segment is in its own video. It is a false positive with probability equal to its stated uncertainty."""
    rng = np.random.default_rng(seed)
    stated = rng.random(n)
    wrong = rng.random(n) < stated
    truth = gt([(f'v{i}', 10, 20, 0) for i in range(n)])
    prediction = pred([(f'v{i}', 30, 40, 0, .5) if bad else (f'v{i}', 10, 20, 0, .5) for i, bad in enumerate(wrong)])
    return truth, prediction, stated


def test_segment_uncertainty_is_checked_against_false_positives_per_decile():
    truth, prediction, stated = _segments_with_known_errors()
    got = lr.validate_segment_uncertainty(truth, prediction, stated, split='validation')
    assert got['calibrated'] and got['unit'] == 'segment' and got['split'] == 'validation' and len(got['table']['bins']) == 10
    shrunk = lr.validate_segment_uncertainty(truth, prediction, stated ** 3)
    assert not shrunk['calibrated'] and shrunk['ranks_errors']


def test_video_uncertainty_uses_label_errors():
    scores = np.array([[.9, .1], [.2, .8], [.7, .3]])
    labels = np.array([[1, 0], [1, 0], [1, 1]])
    assert lr.video_label_errors(scores, labels).tolist() == [0, 1, 0]
    got = lr.validate_video_uncertainty([.1, .9, .2], scores, labels)
    assert got['unit'] == 'video' and got['n'] == 3


def test_report_cli(tmp_path):
    truth, prediction, stated = _segments_with_known_errors(n=400)
    truth.to_csv(tmp_path / 'gt.csv', index=False)
    prediction.assign(uncertainty=stated).to_csv(tmp_path / 'p.csv', index=False)
    assert lr.main(['--ground_truth', str(tmp_path / 'gt.csv'), '--predictions', str(tmp_path / 'p.csv'), '--out', str(tmp_path / 'r'), '--uncertainty_column', 'uncertainty']) == 0
    report = json.loads((tmp_path / 'r' / 'localization_report.json').read_text())
    assert 'avg_0.1_0.7' in report['detection']['averages'] and report['uncertainty']['unit'] == 'segment'
    assert report['detection']['map_per_threshold']['0.5'] == pytest.approx(_ap(truth, prediction, [0.5])[0])
    assert pd.read_csv(tmp_path / 'r' / 'per_class_ap.csv')['0.5'].iloc[0] == pytest.approx(_ap(truth, prediction, [0.5])[0])


def test_applied_to_is_required():
    with pytest.raises(TypeError, match='applied_to'):
        lr.attach_expected_error(pd.DataFrame({'a': [0]}), [.1], {'calibrated': True, 'unit': 'segment', 'split': 'validation'})


def test_a_check_with_no_named_split_cannot_attach():
    with pytest.raises(ValueError, match='does not name'):
        lr.attach_expected_error(pd.DataFrame({'a': [0]}), [.1], {'calibrated': True, 'unit': 'segment'}, applied_to='test')


def test_a_video_level_check_cannot_attach_to_segments():
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t)
    video = lr.validate_uncertainty(stated, wrong, unit='video', split='validation')
    assert video['calibrated']
    table = pd.DataFrame({'t-start': np.zeros(len(stated))})
    with pytest.raises(ValueError, match='per video'):
        lr.attach_expected_error(table, stated, video, applied_to='test', unit='segment')
    assert 'expected_error' in lr.attach_expected_error(table, stated, video, applied_to='test', unit='video').columns


@pytest.mark.parametrize('stated,wrong', [([.0], [0.]), ([.02] * 5, [0.] * 5), ([.3] * 99, [0., 0., 1.] * 33)])
def test_too_few_rows_are_never_called_calibrated(stated, wrong):
    got = lr.validate_uncertainty(np.array(stated), np.array(wrong))
    assert not got['calibrated'] and any('under the minimum' in reason for reason in got['calibration_failures'])


def test_one_terrible_decile_fails_even_with_a_small_weighted_gap():
    """900 rows stated .2, right about that; 100 rows stated .05, half of them wrong: weighted gap .045, decile gap .45."""
    stated = np.r_[np.full(900, .2), np.full(100, .05)]
    wrong = np.r_[(np.arange(900) % 5 == 0) * 1., (np.arange(100) % 2 == 0) * 1.]
    got = lr.validate_uncertainty(stated, wrong)
    assert got['calibration_error'] <= .05
    assert not got['calibrated'] and any('decile' in reason for reason in got['calibration_failures'])


def test_honest_small_samples_rarely_fail_the_decile_rule():
    """Truly calibrated data at n = 150, with the weighted-gap rule set aside: the joint interval should fail about 5% of the time."""
    failed = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        stated = rng.random(150)
        wrong = (rng.random(150) < stated) * 1.
        failed += not lr.validate_uncertainty(stated, wrong, max_gap=1.0)['calibrated']
    assert failed / 200 <= .10


def test_a_terrible_decile_fails_at_the_minimum_size_too():
    stated = np.r_[np.full(90, .2), np.full(10, .05)]  # the lowest decile: stated .05, half wrong, 10 rows
    wrong = np.r_[(np.arange(90) % 5 == 0) * 1., (np.arange(10) % 2 == 0) * 1.]
    got = lr.validate_uncertainty(stated, wrong)
    assert got['n'] == 100 and not got['calibrated'] and any('decile' in reason for reason in got['calibration_failures'])


def test_nan_uncertainty_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        lr.validate_uncertainty(np.array([.1, np.nan, .2]), np.array([0., 1., 1.]))
    validation = {'calibrated': True, 'unit': 'segment', 'split': 'validation'}
    with pytest.raises(ValueError, match='finite'):
        lr.attach_expected_error(pd.DataFrame({'a': [0, 1]}), [.1, np.nan], validation, applied_to='test')


def _honest_verdicts(n, seeds=100):
    verdicts = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        stated = rng.random(n)
        verdicts.append(lr.validate_uncertainty(stated, (rng.random(n) < stated) * 1.)['verdict'])
    return pd.Series(verdicts)


def test_honest_data_needs_hundreds_of_rows_to_pass_the_default_rule():
    """The real behaviour, with no rule set aside: calibrated data rarely passes at 100 rows and usually does at 800."""
    passed = {n: _honest_verdicts(n).eq('calibrated').mean() for n in (100, 300, 800)}
    assert passed[100] <= .10 and .15 <= passed[300] <= .60 and passed[800] >= .80


def test_a_failure_of_honest_small_samples_reads_inconclusive_not_miscalibrated():
    verdicts = _honest_verdicts(100)
    assert verdicts.eq('inconclusive').mean() >= .75 and verdicts.eq('miscalibrated').mean() <= .20


def test_verdicts_for_too_few_rows_and_real_miscalibration():
    few = lr.validate_uncertainty(np.full(99, .3), np.array([0., 0., 1.] * 33))
    assert not few['calibrated'] and few['verdict'] == 'inconclusive' and '500 to 800' in few['verdict_note']
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t ** 4)
    big = lr.validate_uncertainty(stated, wrong)
    assert big['verdict'] == 'miscalibrated' and big['verdict_note'] is None and big['calibration_error'] > big['gap_if_calibrated_p95']
    stated = np.r_[np.full(900, .2), np.full(100, .05)]
    wrong = np.r_[(np.arange(900) % 5 == 0) * 1., (np.arange(100) % 2 == 0) * 1.]
    assert lr.validate_uncertainty(stated, wrong)['verdict'] == 'miscalibrated'
    stated, wrong = _simulated_uncertainty(0, lambda t, _: t)
    assert lr.validate_uncertainty(stated, wrong)['verdict'] == 'calibrated'


def test_an_inconclusive_check_attaches_nothing():
    stated = np.full(99, .3)
    check = lr.validate_uncertainty(stated, np.array([0., 0., 1.] * 33), unit='segment', split='validation')
    table = pd.DataFrame({'t-start': np.zeros(99)})
    assert 'expected_error' not in lr.attach_expected_error(table, stated, check, applied_to='test').columns

