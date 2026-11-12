"""Automatic model selection for video (R3): `select_model.py`'s engine.

grid.py       the search grid, its candidates and their deeptune.py command lines (torch-free)
checks.py     every candidate checked by deeptune.py's own parser, then built once to count and size it
sealed.py     the one dev / sealed-test split, made before anything trains
ledger.py     one JSON line per finished candidate, for resuming
runner.py     deeptune.py as a subprocess, with out-of-memory recorded as "did not fit"
estimate.py   the run count and the rough time estimate printed before training
ranking.py    leaderboard, intervals and the paired difference of the top two
search.py     the orchestration, and the search strategies (exhaustive now; R3b adds smart)
"""
