"""Reproducible Oxford-IIIT Pet fine-tuning; downloads only with --download.

Whole-animal boxes use trimap foreground plus boundary (1 and 3); dog images
are negative examples. Official test images never select epochs or thresholds.
This is detection evaluation, not a labelled tracking benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

SOURCE = 'https://www.robots.ox.ac.uk/~vgg/data/pets/'


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'DeepTune academic cat tracking experiment'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(destination)


def prepare(root):
    checksums = {'images': '5c4f3ee8e5d25df40f4fd59a7f44e54c',
                 'annotations': '95a8c909bbe2e81eed6a22bccdf3f68f'}
    for name in ('images', 'annotations'):
        archive = root / (name + '.tar.gz')
        download('https://thor.robots.ox.ac.uk/datasets/pets/' + archive.name, archive)
        with archive.open('rb') as stream:
            if hashlib.file_digest(stream, 'md5').hexdigest() != checksums[name]:
                raise ValueError(f'Checksum mismatch: {archive}; do not extract')
        if not (root / (name + '.extracted')).exists():
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter='data')
            (root / (name + '.extracted')).write_text('complete', encoding='utf-8')


def prepare_subset(args):
    """Download only the reproducible subset from an original-file HF mirror."""
    root = args.data
    mirror = 'https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/main/oxford-iiit-pet/'
    archive = root / 'annotations.tar.gz'
    download(mirror + archive.name, archive)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'md5').hexdigest() != '95a8c909bbe2e81eed6a22bccdf3f68f':
            raise ValueError('Mirror annotations differ from the original Oxford archive')
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    splits = split_records(root, args.seed, args.train_count, args.val_count, args.test_count)
    excluded = set(args.exclude_image)
    known = {row['id'] for part in splits.values() for row in part}
    if excluded - known:
        raise ValueError(f'Excluded images are outside the selected cohort: {sorted(excluded - known)}')
    rows = [row for part in splits.values() for row in part if row['id'] not in excluded]
    (root / 'images').mkdir(exist_ok=True)
    def fetch(row):
        path = root / 'images' / (row['id'] + '.jpg')
        for attempt in range(3):
            try:
                download(mirror + 'images/' + path.name, path)
                with Image.open(path) as image:
                    image.verify()
                return path.name, hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    hashes = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, row) for row in rows]
        for future in as_completed(futures):
            name, digest = future.result()
            hashes[name] = digest
            if len(hashes) % 50 == 0:
                print(f'Images ready: {len(hashes)}/{len(rows)}', flush=True)
    (root / 'mirror_provenance.json').write_text(json.dumps(
        {'mirror': mirror, 'annotations_md5': '95a8c909bbe2e81eed6a22bccdf3f68f',
         'excluded_images': sorted(excluded),
         'image_sha256': hashes}, indent=2))


def mask_box(mask):
    y, x = np.where(np.isin(mask, (1, 3)))
    if len(x) == 0:
        raise ValueError('Pet mask contains no foreground')
    return [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]


def records(root, split):
    result = []
    for line in (root / 'annotations' / (split + '.txt')).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, breed, species, _ = line.split()
        result.append({'id': name, 'cat': species == '1', 'breed': int(breed)})
    return result


def split_records(root, seed, train_count, val_count, test_count):
    rng = random.Random(seed)
    pool = records(root, 'trainval')
    # Stratify by species, with 80% cats and 20% dog negatives in each subset.
    def take(rows, count):
        cats = [r for r in rows if r['cat']]
        dogs = [r for r in rows if not r['cat']]
        rng.shuffle(cats)
        rng.shuffle(dogs)
        selected = cats[:int(count * .8)] + dogs[:count - int(count * .8)]
        if len(selected) != count:
            raise ValueError('Requested subset exceeds available images')
        rng.shuffle(selected)
        return selected
    val = take(pool, val_count)
    val_ids = {r['id'] for r in val}
    train = take([r for r in pool if r['id'] not in val_ids], train_count)
    test = take(records(root, 'test'), test_count)
    return {'train': train, 'val': val, 'test': test}


def validate_splits(root, splits, exclude_images=()):
    """Validate a fixed cohort; exclusions never replace or move its images."""
    excluded = set(exclude_images)
    known = {row['id'] for rows in splits.values() for row in rows}
    if excluded - known:
        raise ValueError(f'Excluded images are outside the selected cohort: {sorted(excluded - known)}')
    result, seen = {}, set()
    for split, rows in splits.items():
        result[split] = []
        for row in rows:
            if row['id'] in seen:
                raise ValueError(f'Image occurs in multiple splits: {row["id"]}')
            seen.add(row['id'])
            if row['id'] in excluded:
                continue
            with Image.open(root / 'images' / (row['id'] + '.jpg')) as image:
                size = image.size
                image.verify()
            if row['cat']:
                with Image.open(root / 'annotations' / 'trimaps' / (row['id'] + '.png')) as mask:
                    if mask.size != size:
                        raise ValueError(f'Image and mask sizes differ: {row["id"]}')
                    try:
                        mask_box(np.asarray(mask))
                    except ValueError as error:
                        raise ValueError(f'Invalid mask for {row["id"]}: {error}') from error
            result[split].append(row)
        if not result[split]:
            raise ValueError(f'Empty split after exclusions: {split}')
    return result


class Pets:
    def __init__(self, root, rows, augment=False):
        self.root, self.rows, self.augment = root, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        with Image.open(self.root / 'images' / (row['id'] + '.jpg')) as image:
            pixels = np.array(image.convert('RGB'))
        boxes = []
        if row['cat']:
            with Image.open(self.root / 'annotations' / 'trimaps' / (row['id'] + '.png')) as mask:
                boxes = [mask_box(np.asarray(mask))]
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < .5:
            tensor = tensor.flip(-1)
            box_tensor[:, [0, 2]] = pixels.shape[1] - box_tensor[:, [2, 0]]
        return tensor, {'boxes': box_tensor, 'labels': torch.ones(len(boxes), dtype=torch.int64)}


def detection_metrics(predictions, targets, threshold=.5):
    from spatial.track import iou_matrix
    scored = []
    total = sum(len(t['boxes']) for t in targets)
    for pred, target in zip(predictions, targets):
        matched = set()
        order = np.argsort(-np.asarray(pred['scores']))
        for i in order:
            candidates = iou_matrix([pred['boxes'][i]], target['boxes'])[0] if len(target['boxes']) else []
            j = max((j for j in range(len(candidates)) if j not in matched),
                    key=lambda j: candidates[j], default=-1)
            correct = j >= 0 and candidates[j] >= .5
            if correct:
                matched.add(j)
            scored.append((float(pred['scores'][i]), int(correct)))
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([v for _, v in scored])
    precision = tp / np.arange(1, len(scored) + 1)
    recall = tp / max(1, total)
    # All-points interpolated AP at IoU .5 (not COCO mAP).
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(np.r_[0, recall]) * envelope))
    accepted = [hit for score, hit in scored if score >= threshold]
    return {'ap50': ap, 'precision_at_0.5': sum(accepted) / max(1, len(accepted)),
            'recall_at_0.5': sum(accepted) / max(1, total), 'ground_truth_cats': total,
            'predictions_at_0.5': len(accepted), 'images': len(targets)}


def evaluate(model, dataset, device):
    import torch
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            predictions.append({key: output[key].cpu().numpy() for key in ('boxes', 'scores')})
            targets.append({'boxes': target['boxes'].numpy()})
    return detection_metrics(predictions, targets)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--mirror-subset', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--train-count', type=int, default=900)
    p.add_argument('--val-count', type=int, default=240)
    p.add_argument('--test-count', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude-image', action='append', default=[], metavar='ID',
                   help='Explicit unavailable image ID; logged, with no replacement or split reassignment.')
    args = p.parse_args()
    if args.mirror_subset:
        prepare_subset(args)
    if args.download:
        prepare(args.data)
    if args.prepare_only:
        return
    if args.epochs < 1:
        p.error('--epochs must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'best.pt').exists():
        p.error('Output already has a checkpoint; choose a new output folder')
    splits = validate_splits(args.data,
                            split_records(args.data, args.seed, args.train_count, args.val_count, args.test_count),
                            args.exclude_image)
    print('Validated image splits: ' + json.dumps({k: len(v) for k, v in splits.items()}), flush=True)
    import torch
    from spatial.pet_detector import make_model, ARCHITECTURE
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested GPU is unavailable')
        torch.cuda.set_per_process_memory_fraction(.55)
    (args.out / 'splits.json').write_text(json.dumps(splits, indent=2))
    provenance = {'dataset': SOURCE, 'dataset_licence': 'CC-BY-SA-4.0; image copyrights remain with owners',
                  'initial_weights': 'torchvision FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1',
                  'weights_terms': 'https://pytorch.org/vision/stable/models.html; COCO source data terms apply',
                  'box_source': 'whole animal trimap labels 1 and 3', 'seed': args.seed,
                  'architecture': ARCHITECTURE, 'num_workers': 0, 'batch_size': 2,
                  'excluded_images': sorted(set(args.exclude_image)),
                  'split_counts': {k: len(v) for k, v in splits.items()},
                  'split_sha256': hashlib.sha256((args.out / 'splits.json').read_bytes()).hexdigest()}
    model = make_model(pretrained=True).to(args.device)
    # Freeze the image backbone; train proposal and detection heads only.
    for param in model.backbone.parameters():
        param.requires_grad = False
    train = Pets(args.data, splits['train'], augment=True)
    val = Pets(args.data, splits['val'])
    test = Pets(args.data, splits['test'])
    loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=0,
                                        collate_fn=lambda rows: tuple(zip(*rows)))
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad], lr=.001,
                                momentum=.9, weight_decay=.0005)
    print('Evaluating pretrained cat baseline...', flush=True)
    report = {'provenance': provenance, 'baseline_val': evaluate(model, val, args.device),
              'baseline_test': evaluate(model, test, args.device), 'epochs': []}
    torch.save({'architecture': ARCHITECTURE, 'labels': ['background', 'cat'],
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'provenance': provenance}, args.out / 'baseline.pt')
    best = -1.
    for epoch in range(args.epochs):
        start = time.monotonic()
        model.train()
        model.backbone.eval()
        losses = []
        for step, (images, targets) in enumerate(loader):
            images = [i.to(args.device) for i in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} loss {losses[-1]:.4f}', flush=True)
        metrics = evaluate(model, val, args.device)
        entry = {'epoch': epoch+1, 'mean_loss': float(np.mean(losses)), 'val': metrics,
                 'seconds': time.monotonic() - start}
        report['epochs'].append(entry)
        print(json.dumps(entry), flush=True)
        if metrics['ap50'] > best:
            best = metrics['ap50']
            report['best_epoch'] = epoch + 1
            torch.save({'architecture': ARCHITECTURE, 'labels': ['background', 'cat'],
                        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'provenance': {**provenance, 'epoch': epoch+1}}, args.out / 'best.pt')
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    saved = torch.load(args.out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    report['trained_test'] = evaluate(model, test, args.device)
    report['limitations'] = 'Single seed, limited photo subsets; no HOTA/IDF1 video ground truth. Pretraining may overlap source images.'
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
