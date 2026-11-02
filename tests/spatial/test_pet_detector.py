import numpy as np
import pytest
import torch

from spatial.pet_detector import PetDetector, make_model, ARCHITECTURE
from scripts.train_cat_detector import mask_box, detection_metrics


def test_whole_body_box_includes_boundary_not_background():
    mask = np.full((12, 17), 2, dtype=np.uint8)
    mask[3:8, 4:11] = 1
    mask[2, 4:11] = 3
    assert mask_box(mask) == [4, 2, 11, 8]
    with pytest.raises(ValueError):
        mask_box(np.full((3, 3), 2))


def test_metrics_penalize_duplicates_and_false_positives():
    predictions = [{'boxes': [[0, 0, 10, 10], [0, 0, 10, 10]], 'scores': [.8, .7]},
                   {'boxes': [[0, 0, 5, 5]], 'scores': [.9]}]
    targets = [{'boxes': [[0, 0, 10, 10]]}, {'boxes': []}]
    metrics = detection_metrics(predictions, targets)
    assert metrics['ap50'] == pytest.approx(.5)
    assert metrics['precision_at_0.5'] == pytest.approx(1/3)
    assert metrics['recall_at_0.5'] == 1
    assert detection_metrics([{'boxes': [], 'scores': []}], [{'boxes': []}])['ap50'] == 0


def test_local_checkpoint_roundtrip_and_bad_metadata(tmp_path):
    model = make_model()
    path = tmp_path / 'cat.pt'
    torch.save({'architecture': ARCHITECTURE, 'labels': ['background', 'cat'],
                'state_dict': model.state_dict()}, path)
    detector = PetDetector.from_pretrained(path, device='cpu')
    assert torch.equal(model.roi_heads.box_predictor.cls_score.weight,
                       detector.model.roi_heads.box_predictor.cls_score.weight)
    output = detector.detect([np.zeros((40, 80, 3), dtype=np.uint8)], ['cat'])
    assert len(output) == 1
    assert all(d.label == 'cat' and 0 <= d.box[2] <= 80 for d in output[0])
    with pytest.raises(ValueError, match='only'):
        detector.detect([np.zeros((40, 80, 3), dtype=np.uint8)], ['dog'])
    torch.save({'architecture': 'wrong', 'labels': ['cat']}, path)
    with pytest.raises(ValueError, match='Unsupported'):
        PetDetector.from_pretrained(path)


def test_local_gui_command_and_cli_validation(tmp_path):
    from desktop.analyse import TrackConfig, build_track_argv, detector_note, track_button_label
    from track import get_parser, resolve_args, tracker_kwargs
    checkpoint = tmp_path / 'cat.pt'
    checkpoint.write_bytes(b'validated when loaded')
    cfg = TrackConfig(video=tmp_path/'cat.mp4', prompts=['cat'], out=tmp_path/'out',
                      detector='pet_fasterrcnn', detector_model=checkpoint)
    argv = build_track_argv(cfg)
    parser = get_parser()
    args = resolve_args(parser.parse_args(argv), parser)
    assert args.detector_model == str(checkpoint)
    assert args.batch_size == 1
    assert tracker_kwargs(args)['fuse_score'] is True
    assert 'no download' in detector_note('pet_fasterrcnn')
    assert 'Download' not in track_button_label('pet_fasterrcnn')
    cfg.prompts = ['dog']
    with pytest.raises(SystemExit):
        resolve_args(parser.parse_args(build_track_argv(cfg)), parser)


def test_installed_detector_discovery_and_default_cli(tmp_path, monkeypatch):
    from desktop import analyse
    from spatial.detect import DEFAULT_MODELS
    from track import get_parser, resolve_args
    checkpoint = tmp_path / 'best.pt'
    monkeypatch.setitem(DEFAULT_MODELS, 'pet_fasterrcnn', str(checkpoint))
    assert analyse.installed_cat_checkpoint() is None
    checkpoint.write_bytes(b'loaded only by subprocess')
    assert analyse.installed_cat_checkpoint() == checkpoint
    parser = get_parser()
    args = resolve_args(parser.parse_args(['--video', 'cat.mp4', '--out', str(tmp_path / 'out'),
                                          '--detector', 'pet_fasterrcnn', '--prompt', 'cat']), parser)
    assert args.detector_model == str(checkpoint)
