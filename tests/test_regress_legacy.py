"""Bit-exact regression guard: pins the legacy reward, cost and robot update by hash."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needed = ["configs/amc_mid.yaml", "configs/mappo_ws_curr.yaml"]
missing = [n for n in needed if not os.path.exists(os.path.join(ROOT, n))]
if missing:
    print(f"FAIL test_regress_legacy: missing {missing}")
    raise SystemExit(1)
r = subprocess.run([sys.executable, "tools/guard.py", "--threads", "1"],
                   cwd=ROOT, capture_output=True, text=True)
out = (r.stdout or "").strip().splitlines()
print(out[-1] if out else (r.stderr or "")[-300:])
raise SystemExit(r.returncode)
