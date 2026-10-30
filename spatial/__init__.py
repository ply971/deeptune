"""Spatial video analysis for DeepTune: zero-shot detection and tracking.

Modules
-------
- :mod:`spatial.detect`   open-vocabulary detectors (Grounding DINO, OWLv2) and a
  scripted :class:`~spatial.detect.FakeDetector` for tests.
- :mod:`spatial.track`    ByteTrack (Zhang et al., ECCV 2022) re-implemented with a
  constant-velocity Kalman filter and Hungarian matching.
- :mod:`spatial.pipeline` video -> detections -> tracks -> summary.
- :mod:`spatial.io`       frame reading and the output files (detections.json,
  MOTChallenge tracks.csv, summary.json).
- :mod:`spatial.draw`     annotated MP4 overlay.

Boxes are always ``(x1, y1, x2, y2)`` in pixels of the original frame unless a
function says otherwise (MOT files use ``(left, top, width, height)``).

This package must stay importable without torch for the pure-numpy parts
(``track``, ``io``, ``draw``); only the transformer backends in ``detect`` import
torch, lazily.
"""
