"""Appearance embeddings ("Re-ID") for tracking.

IoU alone cannot tell two nearby objects apart, and cannot recognise an object
that comes back after it was lost. An appearance embedding per box gives the
tracker a second cue (BoT-SORT's fused IoU/appearance cost) and lets the
offline linker (:mod:`spatial.postprocess`) join a lost track to the one that
starts when the object reappears.

Every embedder implements::

    embed(frame_rgb [H, W, 3] uint8, boxes [[x1, y1, x2, y2], ...]) -> float32 [N, D]

with L2-normalised rows (cosine similarity = dot product). ``N = 0`` gives a
``[0, D]`` array.

Backends
--------
- ``color``: HSV colour histogram of the box's central region (hue x saturation
  plus value), square-rooted (Hellinger) and L2-normalised. Pure OpenCV/numpy,
  no download, fast; separates differently coloured objects, not lookalikes.
- ``cnn``: global-average-pooled ResNet-18 features (ImageNet weights from
  torchvision, BSD-3-Clause; ImageNet terms apply) of each crop. Generic
  appearance features, not a person-Re-ID model, so they work for any prompt
  (cats, cars, lobsters). Downloads about 45 MB once.
- ``osnet``: OSNet-AIN x1.0, a network trained for person re-identification
  (Torchreid's multi-source MSMT17 + DukeMTMC-reID + CUHK03 model, MIT licence;
  those datasets are for research use, so it is never picked by ``auto``).
  People only. 256x128 crops, 512-d. Downloads 8.9 MB once from the Hugging Face
  mirror ``LibreYOLO/LibreReID-osnet`` at a pinned revision; the file's SHA-256
  is checked before it is loaded.

:func:`resolve_reid` turns ``auto`` into ``cnn`` when the ResNet-18 weights are
already in the local torch hub cache, and ``color`` otherwise, so a default run
never starts a download on its own. The choice is recorded with the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

REID_CHOICES = ('auto', 'cnn', 'osnet', 'color', 'none')
CNN_WEIGHTS_FILE = 'resnet18-f37072fd.pth'  # torchvision ResNet18_Weights.IMAGENET1K_V1
OSNET_REPO = 'LibreYOLO/LibreReID-osnet'
OSNET_REVISION = '98f8d8651b6c18772a8b2879f1d740b3278a1a3d'
OSNET_FILE = 'osnet_ain_x1_0.pt'
OSNET_SHA256 = '34c24e98b6b70c8b62480f846fd0d581aa2fd1535bc0276aecf1f10430b731d1'


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _crop_boxes(frame: np.ndarray, boxes: Sequence, shrink: float = 0.0) -> list[np.ndarray]:
    """Integer crops of each box, clipped to the frame; ``shrink`` trims each side by that fraction."""
    h, w = frame.shape[:2]
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
        c1, r1 = int(max(0, np.floor(x1 + dx))), int(max(0, np.floor(y1 + dy)))
        c2, r2 = int(min(w, np.ceil(x2 - dx))), int(min(h, np.ceil(y2 - dy)))
        if c2 <= c1 or r2 <= r1:  # degenerate: one pixel at the clipped centre
            cx = int(min(max((x1 + x2) / 2, 0), w - 1))
            cy = int(min(max((y1 + y2) / 2, 0), h - 1))
            c1, r1, c2, r2 = cx, cy, cx + 1, cy + 1
        crops.append(frame[r1:r2, c1:c2])
    return crops


class ColorHistEmbedder:
    """HSV histogram embedding (see module docstring). Deterministic, torch-free."""

    name = 'color'

    def __init__(self, hue_bins: int = 16, sat_bins: int = 4, val_bins: int = 8, shrink: float = 0.1):
        self.hue_bins, self.sat_bins, self.val_bins = int(hue_bins), int(sat_bins), int(val_bins)
        self.shrink = float(shrink)
        self.dim = self.hue_bins * self.sat_bins + self.val_bins

    def embed(self, frame: np.ndarray, boxes: Sequence) -> np.ndarray:
        import cv2
        boxes = list(boxes)
        if not boxes:
            return np.zeros((0, self.dim), np.float32)
        hsv = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2HSV)
        out = []
        for crop in _crop_boxes(hsv, boxes, self.shrink):
            crop = np.ascontiguousarray(crop)
            hs = cv2.calcHist([crop], [0, 1], None, [self.hue_bins, self.sat_bins], [0, 180, 0, 256]).ravel()
            v = cv2.calcHist([crop], [2], None, [self.val_bins], [0, 256]).ravel()
            n = max(crop.shape[0] * crop.shape[1], 1)
            out.append(np.sqrt(np.r_[hs, v] / n))
        return _normalise(np.stack(out))

    def describe(self) -> dict:
        return {'name': self.name, 'dim': self.dim, 'bins': [self.hue_bins, self.sat_bins, self.val_bins]}


class CnnEmbedder:
    """ResNet-18 (ImageNet) crop features, 512-d. ``model`` may be injected (tests)."""

    name = 'cnn'

    def __init__(self, device: str = 'auto', size: int = 128, batch_size: int = 32, model=None):
        import torch
        from spatial.detect import pick_device
        self.device = pick_device(device)
        self.size = int(size)
        self.batch_size = int(batch_size)
        if model is None:
            from torchvision.models import ResNet18_Weights, resnet18
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()
        self.model = model.to(self.device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.dim = None

    def _resize(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(np.ascontiguousarray(crop), (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def embed(self, frame: np.ndarray, boxes: Sequence) -> np.ndarray:
        import torch
        boxes = list(boxes)
        if not boxes:
            return np.zeros((0, self.dim or 512), np.float32)
        crops = [self._resize(c) for c in _crop_boxes(frame, boxes)]
        feats = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                x = torch.from_numpy(np.stack(crops[start:start + self.batch_size])).to(self.device)
                x = x.permute(0, 3, 1, 2).float().div(255)
                x = (x - self._mean) / self._std
                feats.append(self.model(x).float().flatten(1).cpu().numpy())
        out = _normalise(np.concatenate(feats))
        self.dim = out.shape[1]
        return out

    def describe(self) -> dict:
        return {'name': self.name, 'backbone': 'resnet18', 'weights': 'IMAGENET1K_V1',
                'licence': 'BSD-3-Clause (torchvision); ImageNet terms apply', 'device': self.device,
                'crop_size': self.size}


def osnet_weights(download: bool = True) -> Path:
    """Path of the pinned OSNet-AIN weights, downloading them if needed; refuses a file with the wrong SHA-256."""
    import hashlib
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(OSNET_REPO, OSNET_FILE, revision=OSNET_REVISION, local_files_only=not download))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OSNET_SHA256:
        raise RuntimeError(f'{path} has SHA-256 {digest}, expected {OSNET_SHA256}; delete it and download again')
    return path


class OsnetEmbedder(CnnEmbedder):
    """OSNet-AIN x1.0 person Re-ID features (see module docstring), 512-d. ``model`` may be injected (tests)."""

    name = 'osnet'

    def __init__(self, device: str = 'auto', size=(256, 128), batch_size: int = 32, model=None):
        if model is None:
            from spatial.osnet import load_osnet_ain
            model = load_osnet_ain(osnet_weights())
        super().__init__(device=device, size=128, batch_size=batch_size, model=model)
        self.height, self.width = (int(v) for v in size)

    def _resize(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(np.ascontiguousarray(crop), (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def describe(self) -> dict:
        return {'name': self.name, 'backbone': 'osnet_ain_x1_0',
                'weights': f'{OSNET_REPO}@{OSNET_REVISION[:8]}/{OSNET_FILE} (Torchreid MS+D+C)',
                'sha256': OSNET_SHA256, 'licence': 'MIT (Torchreid); trained on research-use Re-ID datasets',
                'device': self.device, 'crop_size': [self.height, self.width]}


def cnn_weights_cached() -> bool:
    """Are torchvision's ResNet-18 ImageNet weights already downloaded? (No torch import.)"""
    import os
    if os.environ.get('TORCH_HOME'):
        hub = Path(os.environ['TORCH_HOME']) / 'hub'
    else:
        xdg = os.environ.get('XDG_CACHE_HOME')
        hub = (Path(xdg) if xdg else Path.home() / '.cache') / 'torch' / 'hub'
    return (hub / 'checkpoints' / CNN_WEIGHTS_FILE).is_file()


def resolve_reid(choice: str) -> str:
    """``auto`` -> ``cnn`` if its weights are cached, else ``color``. Others unchanged."""
    if choice not in REID_CHOICES:
        raise ValueError(f'unknown reid {choice!r}; choose from {REID_CHOICES}')
    if choice == 'auto':
        return 'cnn' if cnn_weights_cached() else 'color'
    return choice


def build_embedder(choice: str, device: str = 'auto'):
    """An embedder for a resolved choice (``none`` -> ``None``)."""
    choice = resolve_reid(choice)
    if choice == 'none':
        return None
    if choice == 'color':
        return ColorHistEmbedder()
    if choice == 'osnet':
        return OsnetEmbedder(device=device)
    return CnnEmbedder(device=device)
