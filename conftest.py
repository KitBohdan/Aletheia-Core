"""Project-wide pytest configuration tweaks."""

from __future__ import annotations

import importlib


def _pytest_cov_is_available() -> bool:
    """Return ``True`` when the optional ``pytest-cov`` plugin can be imported."""

    try:
        importlib.import_module("pytest_cov")
    except ModuleNotFoundError:
        return False
    except Exception:
        # Any other exception means the plugin is unusable, so behave as if absent.
        return False
    return True


def pytest_load_initial_conftests(early_config: object, parser: object, args: list[str]) -> None:
    """Strip coverage related CLI flags when ``pytest-cov`` is unavailable.

    The project requests coverage collection via ``pyproject.toml``'s ``addopts``.  The
    execution environment used for kata-style exercises might not have the optional
    ``pytest-cov`` dependency installed, which would normally cause pytest to abort with
    an "unrecognized arguments" error.  By removing the coverage flags before the main
    parsing step we can keep the default developer experience while still allowing the
    test-suite to run in minimal environments.
    """

    if _pytest_cov_is_available():
        return

    removable_prefixes = ("--cov", "--cov-report")
    args[:] = [arg for arg in args if not arg.startswith(removable_prefixes)]
