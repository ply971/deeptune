import argparse
import json
import tarfile

import numpy as np
from PIL import Image
import pytest

from scripts.train_cat_detector import validate_splits


def fixture_cohort(root):
    (root / 'images').mkdir()
    (root / 'annotations' / 'trimaps').mkdir(parents=True)
    splits = {}
    for part in ('train', 'val', 'test'):
        name = part + '_cat'
        Image.new('RGB', (12, 8)).save(root / 'images' / (name + '.jpg'))
        Image.fromarray(np.ones((8, 12), dtype=np.uint8)).save(
            root / 'annotations' / 'trimaps' / (name + '.png'))
        splits[part] = [{'id': name, 'cat': True}]
    return splits


def test_explicit_exclusion_preserves_fixed_evaluation_cohort(tmp_path):
    splits = fixture_cohort(tmp_path)
    splits['train'].append({'id': 'unavailable', 'cat': True})
    with pytest.raises(FileNotFoundError):
        validate_splits(tmp_path, splits)
    validated = validate_splits(tmp_path, splits, ['unavailable'])
    assert validated['train'] == splits['train'][:1]
    assert validated['val'] == splits['val']
    assert validated['test'] == splits['test']
    assert len(splits['train']) == 2
    with pytest.raises(ValueError, match='outside'):
        validate_splits(tmp_path, splits, ['typo'])


def test_validation_rejects_overlap_empty_splits_and_wrong_masks(tmp_path):
    splits = fixture_cohort(tmp_path)
    with pytest.raises(ValueError, match='Empty split'):
        validate_splits(tmp_path, splits, ['train_cat'])
    splits['test'].append(splits['val'][0])
    with pytest.raises(ValueError, match='multiple splits'):
        validate_splits(tmp_path, splits)
    splits['test'].pop()
    mask_path = tmp_path / 'annotations' / 'trimaps' / 'test_cat.png'
    Image.fromarray(np.ones((4, 6), dtype=np.uint8)).save(mask_path)
    with pytest.raises(ValueError, match='sizes differ'):
        validate_splits(tmp_path, splits)


def test_mirror_preparation_does_not_fetch_explicit_exclusions(tmp_path, monkeypatch):
    from scripts import train_cat_detector as training
    splits = fixture_cohort(tmp_path)
    splits['train'].append({'id': 'unavailable', 'cat': True})
    archive = tmp_path / 'annotations.tar.gz'
    with tarfile.open(archive, 'w:gz'):
        pass
    class OriginalDigest:
        def hexdigest(self):
            return '95a8c909bbe2e81eed6a22bccdf3f68f'
    monkeypatch.setattr(training.hashlib, 'file_digest', lambda *a: OriginalDigest())
    monkeypatch.setattr(training, 'split_records', lambda *a: splits)
    fetched = []
    monkeypatch.setattr(training, 'download', lambda url, path: fetched.append(path.name))
    args = argparse.Namespace(data=tmp_path, seed=42, train_count=2, val_count=1, test_count=1,
                              exclude_image=['unavailable'])
    training.prepare_subset(args)
    assert sorted(fetched) == ['annotations.tar.gz', 'test_cat.jpg', 'train_cat.jpg', 'val_cat.jpg']
    provenance = json.loads((tmp_path / 'mirror_provenance.json').read_text())
    assert provenance['excluded_images'] == ['unavailable']
    assert set(provenance['image_sha256']) == {'test_cat.jpg', 'train_cat.jpg', 'val_cat.jpg'}
