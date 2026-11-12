"""Appearance embeddings and BoT-SORT's fused IoU/appearance cost."""
import numpy as np
import pytest

from spatial import reid
from spatial.detect import Detection
from spatial.track import ByteTracker


def _frame():
    f = np.zeros((40, 80, 3), np.uint8)
    f[:, :40] = (220, 30, 30)    # red half
    f[:, 40:] = (30, 30, 220)    # blue half
    return f


def test_color_embedding_shape_norm_and_similarity():
    e = reid.ColorHistEmbedder()
    out = e.embed(_frame(), [(0, 0, 20, 20), (10, 10, 30, 30), (50, 0, 70, 20)])
    assert out.shape == (3, e.dim) and out.dtype == np.float32
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(3), abs=1e-5)
    assert out[0] @ out[1] > 0.99      # both red
    assert out[0] @ out[2] < 0.5       # red vs blue
    assert e.embed(_frame(), []).shape == (0, e.dim)


def test_color_embedding_survives_degenerate_and_outside_boxes():
    e = reid.ColorHistEmbedder()
    out = e.embed(_frame(), [(5, 5, 5, 5), (-10, -10, -2, -2), (70, 30, 200, 90)])
    assert out.shape == (3, e.dim) and np.isfinite(out).all()


def test_auto_picks_cnn_only_when_its_weights_are_cached(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_HOME', str(tmp_path))
    assert reid.resolve_reid('auto') == 'color'
    ckpt = tmp_path / 'hub' / 'checkpoints' / reid.CNN_WEIGHTS_FILE
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'x')
    assert reid.resolve_reid('auto') == 'cnn'
    assert reid.resolve_reid('none') == 'none' and reid.resolve_reid('color') == 'color'
    with pytest.raises(ValueError):
        reid.resolve_reid('resnet50')   # unknown names are refused
    assert reid.build_embedder('none') is None
    assert isinstance(reid.build_embedder('color'), reid.ColorHistEmbedder)


def test_cnn_embedder_with_an_injected_model_on_cpu():
    torch = pytest.importorskip('torch')
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    e = reid.CnnEmbedder(device='cpu', size=32, batch_size=2, model=model)
    out = e.embed(_frame(), [(0, 0, 20, 20), (10, 10, 30, 30), (50, 0, 70, 20)])
    assert out.shape == (3, 8)
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(3), abs=1e-5)
    assert e.embed(_frame(), []).shape[0] == 0
    assert e.describe()['backbone'] == 'resnet18'


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


A, B = _unit([1, 0, 0]), _unit([0, 1, 0])


def _crossing(with_appearance):
    """Frame 1: one object (appearance A). Frame 2: two boxes near it; the one with the
    better IoU looks like B, the slightly worse one looks like A."""
    tr = ByteTracker(high_thresh=0.5)
    first = [Detection((0, 0, 10, 10), 0.9, 'x')]
    tr.update(first, embeddings=[A] if with_appearance else None)
    second = [Detection((1, 0, 11, 10), 0.9, 'x'), Detection((2, 0, 12, 10), 0.9, 'x')]
    out = tr.update(second, embeddings=[B, A] if with_appearance else None)
    return {t.track_id: t.box for t in out}


def test_iou_alone_follows_the_closer_box():
    out = _crossing(False)
    assert out[1] == pytest.approx((1, 0, 11, 10), abs=0.6)


def test_appearance_keeps_the_identity_on_the_lookalike():
    out = _crossing(True)
    assert out[1] == pytest.approx((2, 0, 12, 10), abs=0.6)


def test_appearance_cannot_join_boxes_that_are_far_apart():
    # Same appearance but no overlap: proximity gate keeps it a new object.
    tr = ByteTracker(high_thresh=0.5, match_thresh=0.8)
    tr.update([Detection((0, 0, 10, 10), 0.9, 'x')], embeddings=[A])
    out = tr.update([Detection((40, 0, 50, 10), 0.9, 'x')], embeddings=[A])
    assert 1 not in {t.track_id for t in out}


def test_embeddings_must_match_detections_and_thresholds_are_checked():
    tr = ByteTracker()
    with pytest.raises(ValueError):
        tr.update([Detection((0, 0, 10, 10), 0.9, 'x')], embeddings=[A, B])
    for kw in ({'appearance_thresh': 1.5}, {'proximity_thresh': 0}, {'feat_alpha': 1.0}):
        with pytest.raises(ValueError):
            ByteTracker(**kw)


def test_track_appearance_is_a_normalised_moving_average():
    tr = ByteTracker(high_thresh=0.5)
    tr.update([Detection((0, 0, 10, 10), 0.9, 'x')], embeddings=[A])
    tr.update([Detection((0, 0, 10, 10), 0.9, 'x')], embeddings=[B])
    track = tr.tracked[0]
    want = _unit(0.9 * A + 0.1 * B)
    assert track.feat == pytest.approx(want)
    assert track.curr_feat == pytest.approx(B)


def test_track_outputs_report_birth_and_matched_feature():
    tr = ByteTracker(high_thresh=0.5)
    tr.update([Detection((0, 0, 10, 10), 0.9, 'x')])                       # first frame: active at once
    tr.update([Detection((0, 0, 10, 10), 0.9, 'x'), Detection((30, 0, 40, 10), 0.8, 'x')])  # new, unconfirmed
    out = {t.track_id: t for t in tr.update([Detection((0, 0, 10, 10), 0.9, 'x'),
                                             Detection((31, 0, 41, 10), 0.8, 'x')], embeddings=[A, B])}
    assert out[2].start_step == 2 and out[2].birth_box == pytest.approx((30, 0, 40, 10))
    assert out[2].birth_score == pytest.approx(0.8)
    assert out[2].feat == pytest.approx(B)


# --- OSNet-AIN person Re-ID -------------------------------------------------------------------------------


def test_osnet_is_a_choice_but_never_picked_by_auto(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_HOME', str(tmp_path))
    assert 'osnet' in reid.REID_CHOICES
    assert reid.resolve_reid('osnet') == 'osnet'
    assert reid.resolve_reid('auto') in ('cnn', 'color')


def test_osnet_network_gives_512d_features_and_loads_its_own_state_strictly(tmp_path):
    torch = pytest.importorskip('torch')
    from spatial.osnet import OSNetAIN, load_osnet_ain
    torch.manual_seed(0)
    net = OSNetAIN().eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.inference_mode():
        y = net(x)
    assert y.shape == (2, 512) and bool((y >= 0).all())          # Linear + BatchNorm1d + ReLU, as upstream
    # The released checkpoints keep a classifier head; it is dropped, everything else must match exactly.
    state = dict(net.state_dict())
    state['classifier.weight'] = torch.zeros(10, 512)
    state['classifier.bias'] = torch.zeros(10)
    torch.save(state, tmp_path / 'w.pt')
    with torch.inference_mode():
        assert torch.equal(load_osnet_ain(tmp_path / 'w.pt')(x), y)
    del state['conv5.bn.running_mean']
    torch.save(state, tmp_path / 'bad.pt')
    with pytest.raises(RuntimeError):
        load_osnet_ain(tmp_path / 'bad.pt')


def test_osnet_weights_with_the_wrong_checksum_are_refused(tmp_path, monkeypatch):
    pytest.importorskip('huggingface_hub')
    import huggingface_hub
    fake = tmp_path / reid.OSNET_FILE
    fake.write_bytes(b'not the weights')
    calls = []

    def fake_download(repo, filename, revision=None, local_files_only=False):
        calls.append((repo, filename, revision, local_files_only))
        return str(fake)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    with pytest.raises(RuntimeError, match='SHA-256'):
        reid.osnet_weights(download=False)
    assert calls == [(reid.OSNET_REPO, reid.OSNET_FILE, reid.OSNET_REVISION, True)]   # pinned revision


def test_osnet_embedder_feeds_256x128_crops_and_normalises():
    torch = pytest.importorskip('torch')
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append(tuple(x.shape))
            return x.mean(dim=(2, 3))      # [B, 3]: mean normalised colour

    e = reid.OsnetEmbedder(device='cpu', batch_size=2, model=Probe())
    out = e.embed(_frame(), [(0, 0, 20, 20), (10, 10, 30, 30), (50, 0, 70, 20)])
    assert seen == [(2, 3, 256, 128), (1, 3, 256, 128)]
    assert out.shape == (3, 3) and np.linalg.norm(out, axis=1) == pytest.approx(np.ones(3), abs=1e-5)
    assert out[0] @ out[1] > 0.99 and out[0] @ out[2] < out[0] @ out[1]
    assert e.describe()['name'] == 'osnet' and e.describe()['sha256'] == reid.OSNET_SHA256
