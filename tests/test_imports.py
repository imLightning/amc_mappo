"""Import every module in the repository; catches a broken dependency graph."""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def modules():
    for sub in ("", "envs", "algorithms", "tools", os.path.join("tools", "_impl")):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        prefix = sub.replace(os.sep, ".") + "." if sub else ""
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f != "__init__.py" and not f.startswith("_"):
                yield f"{prefix}{f[:-3]}"


def main():
    names = list(modules())
    bad = []
    for m in names:
        try:
            importlib.import_module(m)
        except Exception as exc:                      # noqa: BLE001
            bad.append((m, f"{type(exc).__name__}: {exc}"))
    for name, err in bad:
        print(f"  FAIL {name}: {err}")
    print(f"imports OK: {len(names) - len(bad)} modules, {len(bad)} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
