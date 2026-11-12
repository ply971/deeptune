"""Jobs: one candidate, cross-validated under one protocol, for one stage of a strategy (R3b).

Every strategy (exhaustive, smart, random, bayes) asks the engine for jobs and gets their ledger entries back; it never
runs anything itself. The engine builds each job's command line from the candidate with `grid.candidate_argv(role='cv')`,
on `dev.parquet` only, so no strategy has a path to the sealed test file.

Protocols. A score is only comparable with scores of the same protocol, on the same folds:
    cv       the ranking protocol: grouped k-fold through deeptune.py --cv_only, the search's --epochs, early stopping.
             Only its scores pick the winner. Its ledger keys are R3a's, so R3a folders resume.
    screen   the same, with fewer epochs (smart's screening). Never ranked against cv.
    tune     the same, with --tune_epochs (smart's tuning) when they differ from --epochs. Never ranked against cv.
    probe    a frozen backbone's cached snippet features (extract_features.py) and an attentive probe
             (train_feature_head.py) on the same folds (probe.py). Never ranked against cv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

RANKING = 'cv'


@dataclass(frozen=True)
class Protocol:
    name: str = RANKING
    method: str = 'deeptune_cv'  # or 'feature_probe'
    epochs: Optional[int] = None  # None: the search's --epochs
    details: tuple = ()  # (name, value) pairs that change the result, for the ledger key

    @property
    def ranking(self) -> bool:
        return self.name == RANKING

    def key_settings(self, training: dict) -> dict:
        """The training settings this protocol's ledger keys hash. The ranking protocol keeps R3a's keys unchanged."""
        if self.ranking:
            return training
        return {**training, 'protocol': self.name, 'method': self.method,
                'epochs': training['epochs'] if self.epochs is None else self.epochs, **dict(self.details)}

    def describe(self, epochs: int) -> str:
        if self.method == 'feature_probe':
            return 'cached frozen features + attentive probe (probe protocol)'
        used = epochs if self.epochs is None else self.epochs
        return f'deeptune.py grouped k-fold, {used} epoch{"s" if used != 1 else ""} ({self.name} protocol)'


@dataclass
class Job:
    item: object  # search.Checked
    protocol: Protocol
    stage: str
    why: str = ''
    key: str = ''
    argv: list = field(default_factory=list)

    @property
    def candidate(self):
        return self.item.candidate
