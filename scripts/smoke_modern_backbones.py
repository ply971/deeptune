"""Real-weights check of the modern backbones, for the GPU gate. Downloads their weights, so it is never run by tests.

    python scripts/smoke_modern_backbones.py --versions vjepa2_vitl siglip2_b16 --frames 8 16

For each version and frame count it loads the weights (a partial or unexpected file stops here), runs a synthetic clip
through the model's own preprocessing, checks the token grid, and reports the time and the peak GPU memory of a frozen
forward. The numbers describe this machine and a synthetic clip; they say nothing about accuracy.

The ``_tiny`` versions need no download, which is how the script itself is tested on CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Load modern backbones with their real weights and measure one frozen forward.')
    p.add_argument('--versions', nargs='+', default=['vjepa2_vitl', 'siglip2_b16'], help='Versions from handlers/video_backbones.py.')
    p.add_argument('--frames', nargs='+', type=int, default=[8, 16], help='Frame counts to try (default 8 16, DeepTune\'s own, per V-014).')
    p.add_argument('--batch_size', type=int, default=1, help='Clips per forward (default 1).')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16', help='Autocast on CUDA (default bf16; ignored on CPU).')
    p.add_argument('--accept_weights_licence', nargs='+', default=None, metavar='LICENCE', help='Needed for opt-in weights.')
    p.add_argument('--out', type=Path, default=None, help='Also write the results to this JSON file.')
    return p


def check_one(version: str, frames: int, batch_size: int, precision: str, backbone=None) -> dict:
    import torch

    from datasets.video_datasets import native_video_transform
    from handlers.video_backbones import hub_backbone
    from src.video.feature_backbones import SnippetBackbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = (backbone or SnippetBackbone(version)).to(device)
    generator = torch.Generator().manual_seed(0)
    clips = [torch.randint(0, 256, (frames, 3, 240, 320), dtype=torch.uint8, generator=generator) for _ in range(batch_size + 1)]
    transform = native_video_transform(version)
    batch = torch.stack([transform(clip) for clip in clips[:batch_size]]).to(device)
    other = transform(clips[-1]).unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and precision == 'bf16')
    began = time.perf_counter()
    with torch.inference_mode(), autocast:
        grid = backbone.token_grid(batch)
    seconds = time.perf_counter() - began
    with torch.inference_mode(), autocast:
        different = backbone.token_grid(other)
    info = hub_backbone(version)
    expected_size = info.input_size // backbone.model.config.patch_size
    if tuple(grid.shape[2:4]) != (expected_size, expected_size):
        raise RuntimeError(f'{version}: spatial grid {tuple(grid.shape[2:4])}, expected {expected_size}x{expected_size}.')
    if not torch.isfinite(grid.float()).all():
        raise RuntimeError(f'{version}: the tokens are not finite.')
    if torch.equal(grid[:1].float(), different.float()):
        raise RuntimeError(f'{version}: two different clips gave identical tokens.')
    return {'version': version, 'frames': frames, 'batch_size': batch_size, 'device': device.type,
            'precision': precision if device.type == 'cuda' else 'fp32', 'weights': backbone.weights_name, 'weights_licence': backbone.weights_licence,
            'token_grid': list(grid.shape[1:4]), 'token_dim': int(grid.shape[-1]), 'parameters': sum(p.numel() for p in backbone.parameters()),
            'seconds_first_forward': round(seconds, 3),
            'peak_gpu_memory_mb': round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.type == 'cuda' else None}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from handlers.video_backbones import accept_weights_licences, hub_backbone
    from src.video.feature_backbones import SnippetBackbone
    accept_weights_licences(args.accept_weights_licence)
    results, failed = [], 0
    for version in args.versions:
        info = hub_backbone(version)
        if info is None:
            raise SystemExit(f'{version} is not a modern backbone; see handlers/video_backbones.py.')
        backbone = SnippetBackbone(version)
        for frames in args.frames:
            if info.fixed_frames is not None and frames != info.fixed_frames:
                print(f'{version}: skipping {frames} frames, it only accepts {info.fixed_frames}.')
                continue
            try:
                result = check_one(version, frames, args.batch_size, args.precision, backbone)
            except RuntimeError as exc:  # includes CUDA out of memory, which is a result here
                failed += 1
                result = {'version': version, 'frames': frames, 'error': str(exc).splitlines()[0]}
                if 'out of memory' in str(exc).lower():
                    import torch
                    torch.cuda.empty_cache()
            print(json.dumps(result))
            results.append(result)
        del backbone
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
