"""Run every ``test_*`` function across tests/ with no GPU and no network.

    python tests/run_all.py

Discovers and calls each ``test_*`` callable in every ``tests/test_*.py`` module. This catches the
files that define only pytest-style functions (no ``__main__`` block) since pytest is not installed in
the project env. Supplies a temp dir to any test whose signature requires ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import tempfile
import traceback

TESTS = pathlib.Path(__file__).parent
sys.path.insert(0, str(TESTS.parent))     # import the package from a plain checkout


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    total = failed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        mod = _load(path)
        for name, fn in sorted(vars(mod).items()):
            if not (name.startswith("test_") and callable(fn) and not inspect.isclass(fn)):
                continue
            total += 1
            kwargs = {}
            params = inspect.signature(fn).parameters
            if "tmp_path" in params and params["tmp_path"].default is inspect.Parameter.empty:
                kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp())
            try:
                fn(**kwargs)
                print(f"  ok  {path.name}::{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {path.name}::{name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
