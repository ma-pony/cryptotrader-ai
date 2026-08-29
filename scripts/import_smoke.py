"""Import every production module without executing a command or opening a venue."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def module_names(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    names = [package.__name__]
    if not hasattr(package, "__path__"):
        return names
    names.extend(info.name for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."))
    return names


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for package_name in ("cryptotrader", "api", "cli"):
        for name in module_names(package_name):
            try:
                importlib.import_module(name)
            except BaseException as error:  # report every module; never hide import failures
                failures.append((name, error))
    if not failures:
        return 0
    for name, error in failures:
        print(f"{name}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
