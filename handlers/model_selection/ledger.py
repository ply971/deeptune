"""`ledger.jsonl`: one JSON line per finished candidate, appended and flushed as each one ends.

A line is written only after its candidate has finished, in one write, then flushed to disk, so Ctrl+C leaves at most an
unfinished candidate that has no line (and runs again on resume). A torn last line from a crash is skipped with a warning.

The key of a line is a hash of the candidate's canonical config, the development data's fingerprint and the training
settings, so a line can never be reused for other data or settings. The last line of a key wins.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ('done', 'failed', 'did_not_fit', 'refused', 'skipped')
FINISHED = ('done', 'failed', 'did_not_fit')  # trained or tried; refused and skipped are decided again on every run
LEDGER_FILE = 'ledger.jsonl'


def candidate_key(config: dict, data_fingerprint: str, training: dict) -> str:
    text = json.dumps({'candidate': config, 'data': data_fingerprint, 'training': training}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.skipped_lines = 0

    def load(self) -> dict:
        """key -> the last entry of that key."""
        entries, self.skipped_lines = {}, 0
        if not self.path.is_file():
            return entries
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entries[entry['key']] = entry
            except (ValueError, KeyError, TypeError):
                self.skipped_lines += 1
        if self.skipped_lines:
            print(f'Warning: {self.skipped_lines} unreadable line(s) in {self.path} were skipped (an interrupted write); '
                  'their candidates count as not run.')
        return entries

    def append(self, entry: dict) -> dict:
        if entry.get('status') not in STATUSES:
            raise ValueError(f'Unknown ledger status {entry.get("status")!r}.')
        entry = {**entry, 'recorded_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        line = json.dumps(entry, sort_keys=True, default=str) + '\n'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size:
            with open(self.path, 'rb') as stream:  # a torn last line must not swallow this one
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    line = '\n' + line
        with open(self.path, 'a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return entry
