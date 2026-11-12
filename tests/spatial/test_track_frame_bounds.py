"""Tracked boxes stay inside the frame, though the Kalman state can run past its edge (seen on a real UCF101 clip)."""
import csv

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from spatial.detect import Detection
from spatial.pipeline import track_video

WIDTH, HEIGHT = 64, 48


class _MovingToTheEdge:
    """Detects a box that moves right quickly and then stays clipped at the frame edge."""
    batch_size = 2

    def describe(self):
        return {'name': 'moving_to_the_edge'}

    def detect(self, frames, prompts, frame_indices=None):
        results = []
        for index in frame_indices:
            left = min(4 * index, WIDTH - 16)
            results.append([Detection((float(left), 8.0, min(left + 16.0, float(WIDTH)), 24.0), 0.9, prompts[0])])
        return results


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (WIDTH, HEIGHT))
    for _ in range(20):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path


def test_written_boxes_stay_inside_the_frame(clip, tmp_path):
    result = track_video(clip, ['thing'], _MovingToTheEdge(), tmp_path / 'out', log=lambda *_: None)
    with open(result['files']['tracks'], newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows
    numeric = [row for row in rows if row[0].lstrip('-').isdigit()]
    assert numeric
    for row in numeric:
        left, top, box_w, box_h = (float(value) for value in row[2:6])
        assert left >= 0 and top >= 0
        assert left + box_w <= WIDTH + 0.01 and top + box_h <= HEIGHT + 0.01
