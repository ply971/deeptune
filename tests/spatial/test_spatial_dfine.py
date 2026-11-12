"""D-FINE backend, COCO prompt mapping and --detector auto. No weights are downloaded."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import track
from spatial.detect import (COCO_CLASSES, DEFAULT_MODELS, FakeDetector, auto_detector, check_coco_prompts,
                            coco_class)


@pytest.mark.parametrize('prompt,want', [
    ('cat', 'cat'), ('Cats', 'cat'), ('kittens', 'cat'), ('people', 'person'), ('pedestrians', 'person'),
    ('Person', 'person'), ('buses', 'bus'), ('traffic lights', 'traffic light'), ('cell-phone', 'cell phone'),
    ('puppy', 'dog'), ('lobster', None), ('St. Bernard', None), ('', None),
])
def test_coco_class_mapping(prompt, want):
    assert coco_class(prompt) == want


def test_coco_list_is_the_80_classes():
    assert len(COCO_CLASSES) == 80 == len(set(COCO_CLASSES))
    assert COCO_CLASSES[0] == 'person' and COCO_CLASSES[15] == 'cat' and COCO_CLASSES[79] == 'toothbrush'


def test_check_coco_prompts_and_auto_choice():
    assert check_coco_prompts(['kittens', 'car']) == {'cat': 'kittens', 'car': 'car'}
    with pytest.raises(ValueError, match='lobster'):
        check_coco_prompts(['cat', 'lobster'])
    with pytest.raises(ValueError, match='both mean'):
        check_coco_prompts(['cat', 'kitten'])
    assert auto_detector(['person', 'car']) == 'dfine'
    assert auto_detector(['person', 'lobster']) == 'grounding_dino'
    assert auto_detector(['cat', 'kitten']) == 'grounding_dino'


class _ScriptedDFine:
    """Stands in for DFineForObjectDetection: 4 queries with fixed classes and boxes."""

    def __init__(self):
        import torch
        self.config = SimpleNamespace(id2label={i: n for i, n in enumerate(COCO_CLASSES)})
        logits = torch.full((1, 4, 80), -10.0)
        for q, (cls, logit) in enumerate([(15, 3.0), (16, 2.0), (0, 1.0), (15, -4.0)]):
            logits[0, q, cls] = logit
        self.logits = logits
        # normalised (cx, cy, w, h)
        self.boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.2, 0.2],
                                    [0.75, 0.5, 0.1, 0.4], [0.1, 0.1, 0.1, 0.1]]])

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values, **kw):
        b = pixel_values.shape[0]
        return SimpleNamespace(logits=self.logits.expand(b, -1, -1), pred_boxes=self.boxes.expand(b, -1, -1))


def test_dfine_backend_keeps_only_the_prompted_classes_with_pixel_boxes():
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from spatial.detect import DFineDetector
    proc = transformers.RTDetrImageProcessor(size={'height': 64, 'width': 64})
    det = DFineDetector(_ScriptedDFine(), proc, device='cpu', box_threshold=0.1, batch_size=2, nms_iou=None)
    frames = np.zeros((3, 100, 200, 3), np.uint8)
    out = det.detect(frames, ['kittens', 'people'])
    assert len(out) == 3
    for dets in out:
        labels = [d.label for d in dets]
        assert labels == ['kittens', 'people']  # dog dropped; the 0.018-score cat is under the threshold
        cat, person = dets
        assert cat.box == pytest.approx((50, 25, 150, 75), abs=0.5)
        assert person.box == pytest.approx((140, 30, 160, 70), abs=0.5)
        assert cat.score == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-4)
    assert det.describe()['name'] == 'dfine'


@pytest.fixture
def lobster_like(tmp_path, two_squares):
    path = two_squares.write(tmp_path / 'v.mp4', fps=10)
    return path, two_squares


def test_auto_detector_resolves_and_is_recorded(tmp_path, lobster_like):
    video, scene = lobster_like
    table = {f: [d.__class__(d.box, d.score, 'cat') for d in dets] for f, dets in scene.detections().items()}
    track.main(['--video', str(video), '--prompt', 'cat', '--out', str(tmp_path / 'a'), '--reid', 'color'],
               detector=FakeDetector(table))
    saved = json.loads((tmp_path / 'a' / 'track_arguments.json').read_text())
    assert saved['detector'] == 'dfine' and saved['detector_requested'] == 'auto'
    assert saved['detector_model'] == DEFAULT_MODELS['dfine']
    assert (saved['box_threshold'], saved['track_high_thresh'], saved['track_low_thresh']) == (0.1, 0.5, 0.1)
    assert saved['new_track_thresh'] == pytest.approx(0.6) and saved['tracker_settings']['fuse_score'] is True
    s = json.loads((tmp_path / 'a' / 'v' / 'summary.json').read_text())
    assert s['labels']['cat']['count'] == 2
    track.main(['--video', str(video), '--prompt', 'lobster', '--out', str(tmp_path / 'b'), '--reid', 'color'],
               detector=FakeDetector(scene.detections()))
    assert json.loads((tmp_path / 'b' / 'track_arguments.json').read_text())['detector'] == 'grounding_dino'


def test_dfine_with_a_non_coco_prompt_is_a_usage_error(tmp_path, lobster_like, capsys):
    video, _ = lobster_like
    with pytest.raises(SystemExit) as exc:
        track.main(['--video', str(video), '--prompt', 'lobster', '--detector', 'dfine', '--out', str(tmp_path / 'o')],
                   detector=FakeDetector({}))
    assert exc.value.code == 2 and 'COCO' in capsys.readouterr().err


@pytest.mark.parametrize('prompt', ['mouse', 'Mice', 'mouses', 'kite', 'kites', 'tie', 'remote', 'keyboard', 'orange'])
def test_auto_sends_ambiguous_coco_words_to_the_open_vocabulary_detector(prompt):
    # Review V-062: a lab-mouse run must not silently become a "computer mouse" detector.
    assert auto_detector([prompt]) == 'grounding_dino'
    assert auto_detector(['person', prompt]) == 'grounding_dino'


@pytest.mark.parametrize('prompt,want', [('computer mouse', 'mouse'), ('necktie', 'tie'), ('remote control', 'remote'),
                                         ('computer keyboard', 'keyboard')])
def test_unambiguous_names_of_those_classes_still_reach_dfine(prompt, want):
    assert coco_class(prompt) == want and auto_detector([prompt]) == 'dfine'


def test_auto_choice_is_explained_printed_and_recorded(tmp_path, lobster_like, capsys):
    from spatial.detect import describe_auto_choice
    assert describe_auto_choice(['people']) == "auto -> dfine: 'people' = COCO 'person'"
    assert 'mouse' in describe_auto_choice(['mouse']) and '--detector dfine' in describe_auto_choice(['mouse'])
    assert 'lobster' in describe_auto_choice(['lobster'])
    video, scene = lobster_like
    track.main(['--video', str(video), '--prompt', 'mouse', '--out', str(tmp_path / 'a'), '--reid', 'color'],
               detector=FakeDetector({}))
    saved = json.loads((tmp_path / 'a' / 'track_arguments.json').read_text())
    assert saved['detector'] == 'grounding_dino' and saved['detector_choice'].startswith('auto -> grounding_dino')
    assert 'auto -> grounding_dino' in capsys.readouterr().out
