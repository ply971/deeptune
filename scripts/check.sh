#!/usr/bin/env bash
# One-command project check: pip check, full pytest suite, video smoke runs.
#
# Usage: scripts/check.sh [--quick]
#   --quick  skip the two video smoke runs (they train tiny models end to end)
#
# Uses $PYTHON if set, else .venv's python when present, else `python`.
# Writes a JUnit report to .artifacts/check/junit.xml and prints a one-line
# summary at the end. Exits non-zero if any step fails.
set -u

quick=0
for arg in "$@"; do
    case "$arg" in
        --quick) quick=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ -n "${PYTHON:-}" ]; then
    python="$PYTHON"
elif [ -x .venv/bin/python ]; then
    python=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    python=.venv/Scripts/python.exe
else
    python=python
fi

report_dir="$root/.artifacts/check"
mkdir -p "$report_dir"
junit="$report_dir/junit.xml"
started=$(date +%s)
failed=0
results=()

device=$("$python" -c "import torch; print(torch.__version__ + ' | ' + (('cuda ' + torch.version.cuda + ' | ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu'))") \
    || device="unknown (torch import failed)"
echo "== device: torch $device"

echo "== pip check"
if "$python" -m pip check; then results+=("pip check: ok"); else results+=("pip check: FAIL"); failed=1; fi

echo "== pytest"
pytest_log="$report_dir/pytest.log"
"$python" -m pytest tests -q --durations=10 -p no:cacheprovider "--junitxml=$junit" | tee "$pytest_log"
pytest_code=${PIPESTATUS[0]}
summary=$(grep -E '[0-9]+ (passed|failed|error)' "$pytest_log" | tail -n 1 | sed -E 's/^=+ *//; s/ *=+$//')
[ -n "$summary" ] || summary="no summary (exit $pytest_code)"
if [ "$pytest_code" -eq 0 ]; then results+=("pytest: $summary"); else results+=("pytest: FAIL: $summary"); failed=1; fi

if [ "$quick" -eq 1 ]; then
    results+=("video smoke: skipped (--quick)")
else
    echo "== smoke resnet18 --manifest"
    if "$python" scripts/smoke_video.py --model resnet18 --manifest; then results+=("smoke resnet18 --manifest: ok"); else results+=("smoke resnet18 --manifest: FAIL"); failed=1; fi
    echo "== smoke r3d_18"
    if "$python" scripts/smoke_video.py --model r3d_18; then results+=("smoke r3d_18: ok"); else results+=("smoke r3d_18: FAIL"); failed=1; fi
    echo "== smoke resnet18 --parquet_bytes"
    if "$python" scripts/smoke_video.py --model resnet18 --parquet_bytes; then results+=("smoke resnet18 --parquet_bytes: ok"); else results+=("smoke resnet18 --parquet_bytes: FAIL"); failed=1; fi
fi

elapsed=$(( $(date +%s) - started ))
status=PASS
[ "$failed" -eq 0 ] || status=FAIL
joined=$(printf '%s; ' "${results[@]}")
echo
echo "CHECK $status | ${joined%; } | device: $device | ${elapsed}s"
exit "$failed"
