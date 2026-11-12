"""Zero-shot (open-vocabulary) object detection on video frames.

Interface
---------
Every detector implements::

    detect(frames, prompts, frame_indices=None) -> list[list[Detection]]

``frames`` is a ``[T, H, W, 3]`` RGB ``uint8`` array (or a list of ``[H, W, 3]``
arrays), ``prompts`` the free-text object names (e.g. ``["lobster", "fish"]``).
The result has one list per frame; each :class:`Detection` holds a pixel box
``(x1, y1, x2, y2)`` in the coordinates of that frame, a score in ``[0, 1]`` and
the prompt it matched (always one of ``prompts``, verbatim after ``strip()``).
``frame_indices`` is optional metadata (the frames' positions in the video);
real backends ignore it, the :class:`FakeDetector` uses it to look up scripted
detections.

Backends
--------
- ``grounding_dino``: Grounding DINO (Liu et al., ECCV 2024), default checkpoint
  ``IDEA-Research/grounding-dino-tiny`` (Apache-2.0).
- ``owlv2``: OWLv2 (Minderer et al., NeurIPS 2023), default checkpoint
  ``google/owlv2-base-patch16-ensemble`` (Apache-2.0).
- ``dfine``: D-FINE (Peng et al., ICLR 2025), a real-time closed-set detector for
  the 80 COCO classes, default checkpoint ``ustc-community/dfine-large-obj2coco-e25``
  (Apache-2.0; Objects365 pre-training, COCO fine-tuning; 57.1 COCO val box AP
  reported by its authors). Prompts must name a COCO class (``person``, ``cat``,
  ``car``...); common synonyms and plurals are mapped (``people`` -> person,
  ``kittens`` -> cat). Much more accurate and faster than the open-vocabulary
  models on those classes, but it cannot find anything else.
- ``auto`` (track.py only) picks ``dfine`` when every prompt is a COCO class and
  ``grounding_dino`` otherwise.

Both run through Hugging Face ``transformers`` processors and their official
post-processing (``post_process_grounded_object_detection``). torch and
transformers are imported lazily so the tracking/IO parts of :mod:`spatial` stay
light.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object: pixel box ``(x1, y1, x2, y2)``, score, prompt label."""

    box: tuple[float, float, float, float]
    score: float
    label: str

    def to_dict(self) -> dict:
        return {'box_xyxy': [round(float(v), 2) for v in self.box],
                'score': round(float(self.score), 4), 'label': self.label}

    @classmethod
    def from_dict(cls, d: Mapping) -> 'Detection':
        return cls(tuple(float(v) for v in d['box_xyxy']), float(d['score']), str(d['label']))


DEFAULT_MODELS = {
    'pet_fasterrcnn': str(Path(__file__).resolve().parents[1] / 'saved_models' / 'cat-detector' / 'best.pt'),
    'grounding_dino': 'IDEA-Research/grounding-dino-tiny',
    'owlv2': 'google/owlv2-base-patch16-ensemble',
    'dfine': 'ustc-community/dfine-large-obj2coco-e25',
}

# The 80 COCO detection classes, in the order of the D-FINE / RT-DETR config (id2label 0..79).
COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush')
COCO_SYNONYMS = {
    'people': 'person', 'pedestrian': 'person', 'human': 'person', 'man': 'person', 'men': 'person',
    'woman': 'person', 'women': 'person', 'child': 'person', 'children': 'person', 'kid': 'person',
    'player': 'person', 'kitten': 'cat', 'kitty': 'cat', 'puppy': 'dog', 'bike': 'bicycle',
    'motorbike': 'motorcycle', 'plane': 'airplane', 'aeroplane': 'airplane', 'sofa': 'couch',
    'television': 'tv', 'tv monitor': 'tv', 'phone': 'cell phone', 'mobile phone': 'cell phone',
    'cellphone': 'cell phone', 'ball': 'sports ball', 'table': 'dining table', 'plant': 'potted plant',
    'doughnut': 'donut', 'hair dryer': 'hair drier', 'fridge': 'refrigerator', 'lorry': 'truck',
    'cattle': 'cow', 'calf': 'cow', 'lamb': 'sheep', 'pony': 'horse', 'laptop computer': 'laptop',
    'computer mouse': 'mouse', 'necktie': 'tie', 'remote control': 'remote', 'tv remote': 'remote',
    'computer keyboard': 'keyboard', 'paper kite': 'kite',
}
# COCO names that usually mean something else in a tracking prompt (a lab mouse, the kite bird, a musical keyboard,
# a tied score...). ``auto`` sends them to the open-vocabulary detector; the unambiguous synonyms above
# (``computer mouse``, ``necktie``...) still reach D-FINE, and ``--detector dfine`` accepts the bare words.
AMBIGUOUS_COCO_WORDS = frozenset({'mouse', 'tie', 'remote', 'kite', 'keyboard', 'orange'})


def coco_class(prompt: str) -> str | None:
    """The COCO class a prompt names (case, plural and common synonyms allowed), else None."""
    p = ' '.join(str(prompt).lower().replace('-', ' ').replace('_', ' ').split())
    candidates = [p]
    if p.endswith('ies'):
        candidates.append(p[:-3] + 'y')
    if p.endswith('es'):
        candidates.append(p[:-2])
    if p.endswith('s'):
        candidates.append(p[:-1])
    for c in candidates:
        if c in COCO_CLASSES:
            return c
        if c in COCO_SYNONYMS:
            return COCO_SYNONYMS[c]
    return None


def check_coco_prompts(prompts: Sequence[str]) -> dict[str, str]:
    """``{coco class: prompt}``; ``ValueError`` if a prompt is not a COCO class or two name the same one."""
    out: dict[str, str] = {}
    unknown = []
    for p in prompts:
        c = coco_class(p)
        if c is None:
            unknown.append(p)
        elif c in out:
            raise ValueError(f'prompts {out[c]!r} and {p!r} both mean the COCO class {c!r}; keep one')
        else:
            out[c] = p
    if unknown:
        raise ValueError(f'D-FINE knows only the 80 COCO classes; not one of them: {unknown}. '
                         'Use --detector grounding_dino or owlv2 for other words.')
    return out


def _is_ambiguous(prompt: str) -> bool:
    p = ' '.join(str(prompt).lower().replace('-', ' ').replace('_', ' ').split())
    singular = {p, p[:-1] if p.endswith('s') else p, p[:-2] if p.endswith('es') else p}
    return bool(singular & AMBIGUOUS_COCO_WORDS)


def auto_detector(prompts: Sequence[str]) -> str:
    """``dfine`` when every prompt is an unambiguous COCO class (and none repeats one), else ``grounding_dino``."""
    if any(_is_ambiguous(p) for p in prompts):
        return 'grounding_dino'
    try:
        check_coco_prompts(prompts)
        return 'dfine'
    except ValueError:
        return 'grounding_dino'


def describe_auto_choice(prompts: Sequence[str]) -> str:
    """One line saying which detector ``auto`` picked for these prompts and why (printed and recorded by track.py)."""
    if auto_detector(prompts) == 'dfine':
        mapping = check_coco_prompts(prompts)
        return 'auto -> dfine: ' + ', '.join(f'{p!r} = COCO {c!r}' for c, p in mapping.items())
    ambiguous = [p for p in prompts if _is_ambiguous(p)]
    if ambiguous:
        return (f'auto -> grounding_dino: {ambiguous} could mean several things (D-FINE would read them as COCO '
                f'{sorted({coco_class(p) for p in ambiguous} - {None})}); use --detector dfine to force that')
    try:
        check_coco_prompts(prompts)
    except ValueError as exc:
        return f'auto -> grounding_dino: {exc}'
    return 'auto -> grounding_dino'

# Score scales differ a lot between the two models (OWLv2 rarely exceeds 0.5),
# so thresholds default per backend. The tracker's high/new-track thresholds are
# derived from these in track.py's CLI (see DETECTOR_TRACK_DEFAULTS there).
DETECTOR_DEFAULTS = {
    'pet_fasterrcnn': {'box_threshold': 0.15, 'text_threshold': None},
    'grounding_dino': {'box_threshold': 0.2, 'text_threshold': 0.25},
    'owlv2': {'box_threshold': 0.1, 'text_threshold': None},
    'dfine': {'box_threshold': 0.1, 'text_threshold': None},
}

LICENCES = {
    'IDEA-Research/grounding-dino-tiny': 'Apache-2.0',
    'google/owlv2-base-patch16-ensemble': 'Apache-2.0',
    'ustc-community/dfine-large-obj2coco-e25': 'Apache-2.0',
}


def clean_prompts(prompts: Sequence[str]) -> list[str]:
    """Strip prompts, drop empties and case-insensitive duplicates (order kept)."""
    out: list[str] = []
    for p in prompts:
        p = str(p).strip()
        if not p:
            continue
        if p.lower() not in {q.lower() for q in out}:
            out.append(p)
    if not out:
        raise ValueError('at least one non-empty prompt is required')
    return out


def grounding_dino_phrase(prompt: str) -> str:
    """The text Grounding DINO sees for a prompt.

    Grounding DINO separates phrases with ``.``, so dots inside a prompt
    ("St. Bernard") are replaced by spaces; the detection label stays the
    original prompt. Raises ``ValueError`` if nothing is left.
    """
    phrase = ' '.join(str(prompt).replace('.', ' ').split()).lower()
    if not phrase:
        raise ValueError(f'prompt {prompt!r} has no text left for Grounding DINO once dots are removed')
    return phrase


def check_grounding_dino_prompts(prompts: Sequence[str]) -> list[str]:
    """Phrases for each prompt; rejects prompts that become identical (e.g. "St. Bernard" / "st bernard")."""
    phrases = [grounding_dino_phrase(p) for p in prompts]
    if len(set(phrases)) != len(phrases):
        raise ValueError(f'prompts {list(prompts)} are identical for Grounding DINO once dots and case are removed')
    return phrases


def cross_label_suppress(dets: Sequence['Detection'], iou_thr: float | None) -> list['Detection']:
    """Class-agnostic NMS: of boxes overlapping by more than ``iou_thr`` keep the
    highest-scoring one, whatever its label. Stops one object found by two prompts
    ("lobster" and "crab") from being counted twice. ``None`` or ``>= 1`` disables."""
    dets = sorted(dets, key=lambda d: -d.score)
    if iou_thr is None or iou_thr >= 1 or len(dets) < 2:
        return list(dets)
    from spatial.track import iou_matrix
    iou = iou_matrix([d.box for d in dets], [d.box for d in dets])
    kept: list[int] = []
    for i in range(len(dets)):
        if all(iou[i, k] <= iou_thr for k in kept):
            kept.append(i)
    return [dets[i] for i in kept]


def pick_device(device: str | None = 'auto') -> str:
    """``'auto'``/``None`` -> ``'cuda'`` when available, else ``'cpu'``."""
    if device in (None, '', 'auto'):
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


def _as_frame_list(frames) -> list[np.ndarray]:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f'frames must be [T, H, W, 3], got {frames.shape}')
        return [frames[i] for i in range(frames.shape[0])]
    out = [np.asarray(f) for f in frames]
    for f in out:
        if f.ndim != 3 or f.shape[-1] != 3:
            raise ValueError(f'each frame must be [H, W, 3], got {f.shape}')
    return out


def finalize_detections(boxes, scores, labels, width: int, height: int,
                        nms_iou: float | None = 0.5, min_size: float = 1.0) -> list[Detection]:
    """Clip boxes to the frame, drop degenerate ones, per-label NMS, sort by score.

    ``nms_iou`` of ``None`` or ``>= 1`` disables NMS. Pure numpy so it is shared by
    every backend and directly testable.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = list(labels)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError('boxes, scores and labels must have the same length')
    if len(boxes) == 0:
        return []
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    ok = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    keep = [i for i in np.flatnonzero(ok)]
    if nms_iou is not None and nms_iou < 1:
        keep = _nms_per_label(boxes, scores, labels, keep, nms_iou)
    keep = sorted(keep, key=lambda i: -scores[i])
    return [Detection(tuple(float(v) for v in boxes[i]), float(scores[i]), labels[i]) for i in keep]


def _nms_per_label(boxes, scores, labels, idx, iou_thr) -> list[int]:
    from spatial.track import iou_matrix
    kept: list[int] = []
    for lab in dict.fromkeys(labels[i] for i in idx):
        cand = sorted((i for i in idx if labels[i] == lab), key=lambda i: -scores[i])
        while cand:
            best, cand = cand[0], cand[1:]
            kept.append(best)
            if cand:
                ious = iou_matrix(boxes[[best]], boxes[cand])[0]
                cand = [c for c, v in zip(cand, ious) if v <= iou_thr]
    return kept


class Detector:
    """Base class. Subclasses implement :meth:`_detect_batch`."""

    name = 'base'

    def __init__(self, batch_size: int = 4, nms_iou: float | None = 0.5):
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self.batch_size = int(batch_size)
        self.nms_iou = nms_iou

    def detect(self, frames, prompts: Sequence[str], frame_indices: Sequence[int] | None = None
               ) -> list[list[Detection]]:
        frames = _as_frame_list(frames)
        prompts = clean_prompts(prompts)
        if frame_indices is not None and len(frame_indices) != len(frames):
            raise ValueError('frame_indices must have one entry per frame')
        out: list[list[Detection]] = []
        for start in range(0, len(frames), self.batch_size):
            chunk = frames[start:start + self.batch_size]
            idx = None if frame_indices is None else list(frame_indices[start:start + self.batch_size])
            out.extend(self._detect_batch(chunk, prompts, idx))
        return out

    def _detect_batch(self, frames: list[np.ndarray], prompts: list[str],
                      frame_indices: list[int] | None) -> list[list[Detection]]:
        raise NotImplementedError

    def describe(self) -> dict:
        """JSON-serialisable description recorded in detections.json."""
        return {'name': self.name}


class FakeDetector(Detector):
    """Scripted detector for tests and dry runs; never looks at pixels.

    ``source`` is either a mapping ``frame_index -> list[Detection]`` (missing
    frames give no detections) or a callable ``(frame_index, frame, prompts) ->
    list[Detection]``. When ``detect`` is called without ``frame_indices`` an
    internal counter numbers frames 0, 1, 2, ... across calls. Detections whose
    label is not among the prompts are dropped, like a real open-vocabulary
    detector that was not asked for them.
    """

    name = 'fake'

    def __init__(self, source: Mapping[int, Sequence[Detection]] | Callable, batch_size: int = 4,
                 nms_iou: float | None = None):
        super().__init__(batch_size=batch_size, nms_iou=nms_iou)
        self.source = source
        self._counter = 0
        self.calls: list[list[int]] = []  # frame indices of each batch, for assertions

    def _detect_batch(self, frames, prompts, frame_indices):
        if frame_indices is None:
            frame_indices = list(range(self._counter, self._counter + len(frames)))
        self._counter = frame_indices[-1] + 1
        self.calls.append(list(frame_indices))
        wanted = {p.lower(): p for p in prompts}
        out = []
        for fi, frame in zip(frame_indices, frames):
            if callable(self.source):
                dets = list(self.source(fi, frame, prompts))
            else:
                dets = list(self.source.get(fi, []))
            h, w = frame.shape[:2]
            dets = [d for d in dets if d.label.lower() in wanted]
            out.append(finalize_detections([d.box for d in dets], [d.score for d in dets],
                                           [wanted[d.label.lower()] for d in dets], w, h,
                                           nms_iou=self.nms_iou))
        return out


class _HFDetector(Detector):
    """Shared plumbing for the transformers backends."""

    def __init__(self, model, processor, device: str | None = 'auto', box_threshold: float = 0.2,
                 batch_size: int = 4, nms_iou: float | None = 0.5, model_id: str | None = None):
        super().__init__(batch_size=batch_size, nms_iou=nms_iou)
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor
        self.box_threshold = float(box_threshold)
        self.model_id = model_id

    def describe(self) -> dict:
        return {'name': self.name, 'model_id': self.model_id, 'device': self.device,
                'box_threshold': self.box_threshold, 'nms_iou': self.nms_iou,
                'licence': LICENCES.get(self.model_id or '', 'see the model card')}

    def _to_device(self, inputs):
        return {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}


class GroundingDinoDetector(_HFDetector):
    """Grounding DINO through ``transformers``.

    Pixel boxes come from the processor's
    ``post_process_grounded_object_detection(outputs, input_ids, threshold=box_threshold,
    text_threshold=..., target_sizes=...)``. Its ``text_labels`` are decoded token
    phrases, which can merge or cut prompts ("lobster fish", "lob"), so instead:

    - **label** = the prompt whose token span holds the highest probability for that
      box (spans found by splitting ``input_ids`` on ``.`` and special tokens);
    - **score** = that probability. The post-processing score is the max over *all*
      text positions, including ``.``/``[SEP]``, which can be high for boxes no
      prompt matches;
    - a box is kept when its score exceeds both ``box_threshold`` and
      ``text_threshold``.

    Prompts may contain dots ("St. Bernard"): see :func:`grounding_dino_phrase`.
    """

    name = 'grounding_dino'

    def __init__(self, model, processor, device='auto', box_threshold: float = 0.2,
                 text_threshold: float = 0.25, batch_size: int = 4, nms_iou=0.5, model_id=None):
        super().__init__(model, processor, device, box_threshold, batch_size, nms_iou, model_id)
        self.text_threshold = float(text_threshold)

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODELS['grounding_dino'], **kw):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        return cls(model, processor, model_id=model_id, **kw)

    def describe(self) -> dict:
        return {**super().describe(), 'text_threshold': self.text_threshold}

    def _prompt_spans(self, input_ids, n_prompts: int) -> list[list[int]]:
        tok = self.processor.tokenizer
        special = set(tok.all_special_ids)
        dot = tok.convert_tokens_to_ids('.')
        spans: list[list[int]] = []
        cur: list[int] = []
        for pos, tid in enumerate(input_ids.tolist()):
            if tid in special or tid == dot:
                if cur:
                    spans.append(cur)
                    cur = []
            else:
                cur.append(pos)
        if cur:
            spans.append(cur)
        if len(spans) != n_prompts:
            raise ValueError(f'could not map {n_prompts} prompts onto the tokenised text '
                             f'({len(spans)} phrases); is the text longer than max_text_len?')
        return spans

    def _detect_batch(self, frames, prompts, frame_indices):
        import torch
        phrases = check_grounding_dino_prompts(prompts)
        text = [list(phrases) for _ in frames]  # processor joins to "a. b."
        inputs = self.processor(images=frames, text=text, return_tensors='pt', padding='longest')
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = [f.shape[:2] for f in frames]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'], threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=sizes)
        probs = torch.sigmoid(outputs.logits.float()).cpu()
        out = []
        for b, (res, (h, w)) in enumerate(zip(results, sizes)):
            spans = self._prompt_spans(inputs['input_ids'][b].cpu(), len(prompts))
            keep = probs[b].max(dim=-1).values > self.box_threshold  # same rule as post-processing
            kept = probs[b][keep]
            boxes = res['boxes'].detach().float().cpu().numpy()
            if len(kept) != len(boxes):
                raise RuntimeError('post-processing kept a different number of boxes than expected')
            if len(kept) == 0:
                out.append([])
                continue
            per_prompt = torch.stack([kept[:, s].max(dim=-1).values for s in spans], dim=1)
            best_p, best_i = per_prompt.max(dim=1)
            scores = best_p.numpy()
            ok = (scores > self.box_threshold) & (scores > self.text_threshold)
            labels = [prompts[int(i)] for i in best_i.tolist()]
            out.append(finalize_detections(boxes[ok], scores[ok], [l for l, k in zip(labels, ok) if k],
                                           w, h, nms_iou=self.nms_iou))
        return out


class Owlv2Detector(_HFDetector):
    """OWLv2 through ``transformers``.

    The OWLv2 image processor pads every frame to a square (bottom/right) before
    resizing, so boxes are rescaled with ``target_sizes=(S, S)``, ``S = max(H, W)``,
    which maps them back to original pixel coordinates; they are then clipped to
    the frame. Labels index directly into the prompt list.
    """

    name = 'owlv2'

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODELS['owlv2'], **kw):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        kw.pop('text_threshold', None)
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
        return cls(model, processor, model_id=model_id, **kw)

    def __init__(self, model, processor, device='auto', box_threshold: float = 0.1,
                 batch_size: int = 4, nms_iou=0.5, model_id=None, text_threshold=None):
        super().__init__(model, processor, device, box_threshold, batch_size, nms_iou, model_id)

    def _detect_batch(self, frames, prompts, frame_indices):
        import torch
        max_len = getattr(self.model.config.text_config, 'max_position_embeddings', 16)
        inputs = self.processor(images=frames, text=[list(prompts) for _ in frames], return_tensors='pt',
                                padding='max_length', max_length=max_len, truncation=True)
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        squares = [(max(f.shape[:2]),) * 2 for f in frames]
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.box_threshold, target_sizes=squares)
        out = []
        for res, f in zip(results, frames):
            h, w = f.shape[:2]
            labels = [prompts[int(i)] for i in res['labels'].cpu().tolist()]
            out.append(finalize_detections(res['boxes'].detach().float().cpu().numpy(),
                                           res['scores'].detach().float().cpu().numpy(), labels, w, h,
                                           nms_iou=self.nms_iou))
        return out


class DFineDetector(_HFDetector):
    """D-FINE (closed-set, 80 COCO classes) through ``transformers``.

    Boxes come from the processor's ``post_process_object_detection`` (top
    queries over all classes, scores in [0, 1]); only the classes named by the
    prompts are kept, each labelled with the user's prompt.
    """

    name = 'dfine'

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODELS['dfine'], **kw):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        kw.pop('text_threshold', None)
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        return cls(model, processor, model_id=model_id, **kw)

    def __init__(self, model, processor, device='auto', box_threshold: float = 0.1,
                 batch_size: int = 4, nms_iou=0.5, model_id=None, text_threshold=None):
        super().__init__(model, processor, device, box_threshold, batch_size, nms_iou, model_id)
        self.id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    def _detect_batch(self, frames, prompts, frame_indices):
        import torch
        wanted = check_coco_prompts(prompts)
        inputs = self._to_device(self.processor(images=frames, return_tensors='pt'))
        with torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = [f.shape[:2] for f in frames]
        results = self.processor.post_process_object_detection(outputs, threshold=self.box_threshold,
                                                                target_sizes=sizes)
        out = []
        for res, (h, w) in zip(results, sizes):
            names = [self.id2label.get(int(i), '') for i in res['labels'].cpu().tolist()]
            keep = [i for i, n in enumerate(names) if n in wanted]
            boxes = res['boxes'].detach().float().cpu().numpy()[keep]
            scores = res['scores'].detach().float().cpu().numpy()[keep]
            out.append(finalize_detections(boxes, scores, [wanted[names[i]] for i in keep], w, h,
                                           nms_iou=self.nms_iou))
        return out


BACKENDS = {'grounding_dino': GroundingDinoDetector, 'owlv2': Owlv2Detector, 'dfine': DFineDetector}


def build_detector(name: str, model_id: str | None = None, **kw) -> Detector:
    """Load a pretrained backend by name (downloads weights on first use)."""
    if name == 'pet_fasterrcnn':
        from spatial.pet_detector import PetDetector
        return PetDetector.from_pretrained(model_id or '', **kw)
    if name not in BACKENDS:
        raise ValueError(f'unknown detector {name!r}; choose from {sorted(BACKENDS)}')
    return BACKENDS[name].from_pretrained(model_id or DEFAULT_MODELS[name], **kw)
