"""Long videos: the online pass keeps appearance vectors only where the offline linker reads them (review V-062)."""
import numpy as np
import pytest

from spatial.detect import Detection
from spatial.pipeline import OnlineTracking
from spatial.postprocess import LinkSettings, PostSettings


def _run(feature_window, steps=200, seed=0):
    """Three objects with their own appearance; each is missed for a stretch, so tracks break and get linked."""
    rng = np.random.default_rng(seed)
    looks = rng.normal(size=(3, 16))
    online = OnlineTracking(10.0, emit_birth=True, feature_window=feature_window)
    for step in range(steps):
        dets, embs = [], []
        for k in range(3):
            if (step // 25 + k) % 4 == 3:          # missing for 25 steps now and then
                continue
            x = 60.0 * k + 0.5 * step
            dets.append(Detection((x, 10.0, x + 20.0, 50.0), 0.9, 'cat'))
            v = looks[k] + rng.normal(0, 0.05, 16)
            embs.append(v / np.linalg.norm(v))
        online.step(step, None, dets, embeddings=np.array(embs) if embs else np.zeros((0, 16)),
                    frame_size=(400, 100))
    return online


def test_bounded_feature_memory_gives_the_same_tracks():
    post = PostSettings(link=LinkSettings(max_gap_s=4.0, feature_window=5))
    full, bounded = _run(None), _run(5)
    kept = [r for r in bounded.rows if 'feat' in r]
    per_track = {}
    for r in kept:
        per_track[r['track_id']] = per_track.get(r['track_id'], 0) + 1
    assert max(per_track.values()) <= 10 and len(kept) < sum('feat' in r for r in full.rows) / 3
    rows_full, stats_full = full.finish(10.0, 1, post)
    rows_bounded, stats_bounded = bounded.finish(10.0, 1, post)
    assert stats_full['links'] > 0 and stats_full == stats_bounded
    assert [(r['frame'], r['track_id'], r['box']) for r in rows_full] == \
           [(r['frame'], r['track_id'], r['box']) for r in rows_bounded]


def test_no_vectors_are_kept_when_nothing_links():
    online = _run(0, steps=40)
    assert not any('feat' in r for r in online.rows)
    with pytest.raises(ValueError):
        OnlineTracking(10.0, feature_window=-1)
