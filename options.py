import argparse
import sys
import torch

from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count


UNIQUE_ID = datetime.now().strftime("%Y%m%d_%H%M") # unique ID based on current date and time (YYYYMMDD_HHMM)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    return torch.device(device_type)

DEVICE = get_device()

CPU_COUNT: int = cpu_count()
NUM_WORKERS: int = min(8, CPU_COUNT // 2) if DEVICE.type != "cpu" else 0
PERSIST_WORK: bool = NUM_WORKERS > 0
PIN_MEM: bool = DEVICE.type == "cuda"

# --- video DataLoader workers (--num_workers, G15) ---
# On Windows every DataLoader worker is a *spawned* process that re-imports
# torch and DeepTune (measured 15-40 s each during the Phase 0 GPU smoke run;
# see scripts/bench_video_io.py for the spawn cost on this machine). Each
# train/val/test/embed loader starts its own pool, so small video datasets are
# faster with no workers at all. 'auto' therefore uses 0 workers on CPU (as
# before), 0 on Windows below WINDOWS_AUTO_MIN_CLIPS clips, and otherwise
# min(WINDOWS_AUTO_MAX_WORKERS, cpu/2) on Windows or min(8, cpu/2) elsewhere.
WINDOWS_AUTO_MIN_CLIPS: int = 500
WINDOWS_AUTO_MAX_WORKERS: int = 4


def resolve_num_workers(requested="auto", n_items=None, device=None, platform=None, cpu=None) -> int:
    """Worker count for a video DataLoader: an int >= 0 is used as given; 'auto' (or None) picks one."""
    if requested is not None and not isinstance(requested, str):
        value = int(requested)
        if value < 0:
            raise ValueError('--num_workers must be 0 or more.')
        return value
    if requested not in (None, 'auto'):
        return resolve_num_workers(int(requested), n_items, device, platform, cpu)
    device = device if device is not None else DEVICE
    platform = platform if platform is not None else sys.platform
    cpu = cpu if cpu is not None else CPU_COUNT
    if getattr(device, 'type', device) == 'cpu':
        return 0
    if platform.startswith('win'):
        if n_items is not None and n_items < WINDOWS_AUTO_MIN_CLIPS:
            return 0
        return max(0, min(WINDOWS_AUTO_MAX_WORKERS, cpu // 2))
    return max(0, min(8, cpu // 2))

# The paths of the Parquet splits of the dataset we are saving/loading from
TRAIN_DATASET_PATH = Path(__file__).parent / f'deeptune_results/train_split_{UNIQUE_ID}.parquet'
VAL_DATASET_PATH = Path(__file__).parent / f'deeptune_results/val_split_{UNIQUE_ID}.parquet'
TEST_DATASET_PATH = Path(__file__).parent / f'deeptune_results/test_split_{UNIQUE_ID}.parquet'

TRAINVAL_OUTPUT_DIR = Path(__file__).parent / f'deeptune_results/output_directory_trainval_{UNIQUE_ID}'
TEST_OUTPUT_DIR = Path(__file__).parent / f'deeptune_results/output_directory_test_{UNIQUE_ID}'
