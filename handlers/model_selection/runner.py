"""Runs deeptune.py for the search, one subprocess per run, one run at a time.

A subprocess frees all of its memory (GPU included) when it ends, so one model cannot leave memory behind for the next.
Its output goes to `deeptune.log` in the run's folder. Out-of-memory errors (GPU or CPU) end as status `did_not_fit`
instead of stopping the search; any other non-zero exit is `failed`, with the last error line as the reason.

A runner is any callable `(argv, run_dir, role) -> RunResult`; the tests inject one that trains nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_FILE = 'deeptune.log'
OUT_OF_MEMORY_MARKERS = ('CUDA out of memory', 'OutOfMemoryError', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_ALLOC_FAILED',
                         'DefaultCPUAllocator: not enough memory', 'paging file is too small', 'MemoryError')


@dataclass
class RunResult:
    status: str  # 'ok', 'failed' or 'did_not_fit'
    seconds: float
    reason: str = ''
    returncode: int = 0


def out_of_memory(text: str) -> bool:
    return any(marker in text for marker in OUT_OF_MEMORY_MARKERS)


def failure_reason(text: str) -> str:
    """The last line that names an error, else the last line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if 'Error' in line or 'error:' in line or 'Exception' in line:
            return line[:500]
    return lines[-1][:500] if lines else 'no output'


def result_from_exit(returncode: int, seconds: float, output: str) -> RunResult:
    if returncode == 0:
        return RunResult('ok', seconds)
    if out_of_memory(output):
        return RunResult('did_not_fit', seconds, f'out of memory: {failure_reason(output)}', returncode)
    return RunResult('failed', seconds, failure_reason(output), returncode)


class SubprocessRunner:
    def __init__(self, python: str = sys.executable, script: Path = REPO / 'deeptune.py', env: dict | None = None):
        self.python, self.script = python, Path(script)
        self.env = dict(os.environ if env is None else env)
        self.env.setdefault('PYTHONIOENCODING', 'utf-8')  # the log is a file: Windows would otherwise encode output as cp1252
        self.env.setdefault('PYTHONUTF8', '1')

    def __call__(self, argv: list, run_dir: Path, role: str) -> RunResult:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / LOG_FILE
        start = time.perf_counter()
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(f'# role: {role}\n# {self.python} {self.script} {subprocess.list2cmdline([str(a) for a in argv])}\n')
            log.flush()
            process = subprocess.Popen([self.python, '-u', str(self.script), *map(str, argv)], stdout=log, stderr=subprocess.STDOUT,
                                       cwd=str(REPO), env=self.env)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                process.kill()
                process.wait()
                raise
        seconds = time.perf_counter() - start
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-20000:]
        return result_from_exit(returncode, seconds, tail)
