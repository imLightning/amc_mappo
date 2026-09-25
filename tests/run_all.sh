#!/usr/bin/env bash
# Run every self-contained check.  PYTHON=<interpreter> to override.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
for t in test_imports test_env_smoke test_regress_legacy; do
  printf '%-22s ' "$t"
  $PY "tests/$t.py" || echo "   (exit $?)"
done
