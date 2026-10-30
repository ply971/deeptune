"""Grounding DINO / OWLv2 backends end-to-end on CPU with tiny random-init models.

No weights are downloaded: models are built from small ``transformers`` configs and
the tokenizers from in-memory vocabularies. Only the output *format* is checked;
random weights say nothing about accuracy.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from spatial.detect import GroundingDinoDetector, Owlv2Detector, pick_device  # noqa: E402

PROMPTS = ['lobster', 'fish']


def tiny_grounding_dino():
    from transformers import (BertConfig, BertTokenizer, GroundingDinoConfig, GroundingDinoForObjectDetection,
                              GroundingDinoImageProcessor, GroundingDinoProcessor, SwinConfig)
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '.', 'lobster', 'fish', 'a', 'red']
    tok = BertTokenizer(vocab={w: i for i, w in enumerate(words)})
    proc = GroundingDinoProcessor(image_processor=GroundingDinoImageProcessor(
        size={'shortest_edge': 64, 'longest_edge': 64}), tokenizer=tok)
    backbone = SwinConfig(image_size=64, patch_size=4, embed_dim=8, depths=[1, 1, 1, 1], num_heads=[1, 1, 1, 1],
                          window_size=2, out_features=['stage2', 'stage3', 'stage4'])
    text = BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=64)
    cfg = GroundingDinoConfig(backbone_config=backbone, text_config=text, num_queries=12, encoder_layers=1,
                              decoder_layers=2, encoder_ffn_dim=16, decoder_ffn_dim=16, encoder_attention_heads=2,
                              decoder_attention_heads=2, d_model=64, max_text_len=32, encoder_n_points=2,
                              decoder_n_points=2)
    torch.manual_seed(0)
    return GroundingDinoForObjectDetection(cfg), proc


def tiny_owlv2():
    from transformers import CLIPTokenizer, Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor, Owlv2Processor
    vocab = {'!': 0, '<|startoftext|>': 1, '<|endoftext|>': 2}
    for w in ['lobster</w>', 'fish</w>', 'a</w>', 'l', 'o', 'b', 's', 't', 'e', 'r', 'f', 'i', 'h']:
        vocab[w] = len(vocab)
    tok = CLIPTokenizer(vocab=vocab, merges=[])
    proc = Owlv2Processor(image_processor=Owlv2ImageProcessor(size={'height': 64, 'width': 64}), tokenizer=tok)
    cfg = Owlv2Config(
        text_config=dict(vocab_size=len(vocab), hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
                         pad_token_id=2),
        vision_config=dict(image_size=64, patch_size=16, hidden_size=16, intermediate_size=16,
                           num_hidden_layers=1, num_attention_heads=2),
        projection_dim=16)
    torch.manual_seed(0)
    return Owlv2ForObjectDetection(cfg), proc


@pytest.fixture(scope='module')
def frames():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (3, 48, 80, 3), dtype=np.uint8)  # non-square on purpose


def check_well_formed(results, frames, n_frames):
    assert isinstance(results, list) and len(results) == n_frames
    h, w = frames.shape[1:3]
    for dets in results:
        assert isinstance(dets, list)
        for d in dets:
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            assert 0.0 <= d.score <= 1.0
            assert d.label in PROMPTS
        assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)


@pytest.mark.parametrize('builder,cls,kw', [
    (tiny_grounding_dino, GroundingDinoDetector, {'box_threshold': 0.0, 'text_threshold': 0.0}),
    (tiny_owlv2, Owlv2Detector, {'box_threshold': 0.0}),
], ids=['grounding_dino', 'owlv2'])
def test_backend_end_to_end_on_cpu(builder, cls, kw, frames):
    model, proc = builder()
    det = cls(model, proc, device='cpu', batch_size=2, nms_iou=None, **kw)
    results = det.detect(frames, PROMPTS)
    check_well_formed(results, frames, len(frames))
    assert sum(len(r) for r in results) > 0  # threshold 0 keeps boxes, so mapping was exercised
    # Batching does not change results (frames are independent).
    single = cls(model, proc, device='cpu', batch_size=1, nms_iou=None, **kw).detect(frames, PROMPTS)
    for a, b in zip(results, single):
        assert [d.label for d in a] == [d.label for d in b]
        assert np.allclose([d.box for d in a], [d.box for d in b], atol=1e-2) if a else not b
    desc = det.describe()
    assert desc['name'] == cls.name and desc['device'] == 'cpu'
    # A list of frames is accepted too, and NMS can only remove boxes.
    nms = cls(model, proc, device='cpu', batch_size=3, nms_iou=0.3, **kw).detect(list(frames[:1]), PROMPTS)
    check_well_formed(nms, frames, 1)
    assert len(nms[0]) <= len(results[0])


def test_backend_threshold_one_returns_nothing(frames):
    model, proc = tiny_owlv2()
    assert Owlv2Detector(model, proc, device='cpu', box_threshold=1.0).detect(frames, PROMPTS) == [[], [], []]
    model, proc = tiny_grounding_dino()
    det = GroundingDinoDetector(model, proc, device='cpu', box_threshold=0.0, text_threshold=1.0)
    assert det.detect(frames[:2], PROMPTS) == [[], []]


def test_grounding_dino_prompt_spans_follow_the_dot_separated_text():
    model, proc = tiny_grounding_dino()
    det = GroundingDinoDetector(model, proc, device='cpu')
    ids = proc(images=[np.zeros((8, 8, 3), np.uint8)], text=[['a red lobster', 'fish']],
               return_tensors='pt')['input_ids'][0]
    spans = det._prompt_spans(ids, 2)
    tok = proc.tokenizer
    assert [tok.decode(ids[s]) for s in spans] == ['a red lobster', 'fish']
    with pytest.raises(ValueError):
        det._prompt_spans(ids, 3)


def test_pick_device():
    assert pick_device('cpu') == 'cpu'
    assert pick_device('auto') in {'cpu', 'cuda'}
    assert pick_device('auto') == ('cuda' if torch.cuda.is_available() else 'cpu')


# --- review V-003: stub models with crafted outputs pin down coordinates and labels ---

class _StubModel(torch.nn.Module):
    """Returns fixed ``logits`` / ``pred_boxes`` (normalised cx, cy, w, h) for every frame."""

    def __init__(self, logits_fn, boxes, config=None):
        super().__init__()
        self.logits_fn = logits_fn
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.config = config
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # so .to(device) has something to move

    def forward(self, input_ids=None, pixel_values=None, **_):
        from types import SimpleNamespace
        b = pixel_values.shape[0]
        logits = self.logits_fn(input_ids, b)
        return SimpleNamespace(logits=logits, pred_boxes=self.boxes.expand(b, -1, -1).clone())


H, W = 48, 80  # non-square: a (h, w) swap or a wrong rescale changes every box


def test_grounding_dino_stub_exact_boxes_labels_and_scores():
    _, proc = tiny_grounding_dino()
    ids = proc(images=[np.zeros((H, W, 3), np.uint8)], text=[PROMPTS], return_tensors='pt')['input_ids'][0]
    tok = proc.tokenizer
    assert tok.convert_ids_to_tokens(ids.tolist()) == ['[CLS]', 'lobster', '.', 'fish', '.', '[SEP]']
    LOB, DOT, FISH, SEP = 1, 2, 3, 5
    rows = [  # (logit per token position, box)
        ({FISH: 3.0}, (0.25, 0.50, 0.25, 0.50)),            # fish
        ({LOB: 2.0}, (0.75, 0.25, 0.10, 0.20)),             # lobster
        ({DOT: 5.0, LOB: 0.5, FISH: -1.0}, (0.50, 0.75, 0.20, 0.25)),  # '.' holds the global max -> lobster
        ({SEP: 4.0, FISH: 1.0}, (0.10, 0.10, 0.10, 0.10)),  # [SEP] holds the global max -> fish
        ({SEP: 4.0}, (0.30, 0.30, 0.10, 0.10)),             # only [SEP] fires: no prompt matches -> dropped
        ({}, (0.60, 0.60, 0.10, 0.10)),                     # nothing fires -> dropped
    ]

    def logits_fn(input_ids, b):
        out = torch.full((b, len(rows), input_ids.shape[1]), -5.0)
        for q, (vals, _) in enumerate(rows):
            for pos, v in vals.items():
                out[:, q, pos] = v
        return out

    model = _StubModel(logits_fn, [box for _, box in rows])
    det = GroundingDinoDetector(model, proc, device='cpu', box_threshold=0.3, text_threshold=0.25,
                                batch_size=2, nms_iou=None)
    frames = np.zeros((3, H, W, 3), np.uint8)
    results = det.detect(frames, PROMPTS)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    want = [('fish', sig(3.0), (10.0, 12.0, 30.0, 36.0)),
            ('lobster', sig(2.0), (56.0, 7.2, 64.0, 16.8)),
            ('fish', sig(1.0), (4.0, 2.4, 12.0, 7.2)),
            ('lobster', sig(0.5), (32.0, 30.0, 48.0, 42.0))]
    for dets in results:
        assert [d.label for d in dets] == [w[0] for w in want]
        assert [d.score for d in dets] == pytest.approx([w[1] for w in want], abs=1e-4)
        for d, w in zip(dets, want):
            assert d.box == pytest.approx(w[2], abs=1e-3)


def test_grounding_dino_stub_dotted_prompt_keeps_original_label():
    _, proc = tiny_grounding_dino()

    def logits_fn(input_ids, b):
        out = torch.full((b, 1, input_ids.shape[1]), -5.0)
        out[:, 0, 1] = 3.0  # first prompt's first token
        return out

    model = _StubModel(logits_fn, [(0.5, 0.5, 0.5, 0.5)])
    det = GroundingDinoDetector(model, proc, device='cpu', box_threshold=0.3, nms_iou=None)
    (dets,) = det.detect(np.zeros((1, H, W, 3), np.uint8), ['Red. Lobster', 'fish'])
    assert [d.label for d in dets] == ['Red. Lobster']
    assert dets[0].box == pytest.approx((20.0, 12.0, 60.0, 36.0), abs=1e-3)


def test_owlv2_stub_exact_boxes_and_labels_on_padded_square():
    from types import SimpleNamespace
    _, proc = tiny_owlv2()
    # OWLv2 boxes are normalised to the padded square S = max(H, W) = 80.
    rows = [((-5.0, 2.0), (0.25, 0.25, 0.25, 0.25)),   # fish
            ((1.0, -5.0), (0.75, 0.40, 0.10, 0.10)),   # lobster
            ((0.5, -5.0), (0.50, 0.55, 0.20, 0.20)),   # lobster, runs past the bottom edge -> clipped to H
            ((-4.0, -4.0), (0.20, 0.20, 0.10, 0.10))]  # below threshold

    def logits_fn(input_ids, b):
        return torch.tensor([r[0] for r in rows], dtype=torch.float32).expand(b, -1, -1).clone()

    cfg = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=16))
    model = _StubModel(logits_fn, [r[1] for r in rows], config=cfg)
    det = Owlv2Detector(model, proc, device='cpu', box_threshold=0.3, batch_size=2, nms_iou=None)
    results = det.detect(np.zeros((3, H, W, 3), np.uint8), PROMPTS)
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    want = [('fish', sig(2.0), (10.0, 10.0, 30.0, 30.0)),
            ('lobster', sig(1.0), (56.0, 28.0, 64.0, 36.0)),
            ('lobster', sig(0.5), (32.0, 36.0, 48.0, 48.0))]
    for dets in results:
        assert [d.label for d in dets] == [w[0] for w in want]
        assert [d.score for d in dets] == pytest.approx([w[1] for w in want], abs=1e-4)
        for d, w in zip(dets, want):
            assert d.box == pytest.approx(w[2], abs=1e-3)
