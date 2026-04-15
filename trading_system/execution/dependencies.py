"""Optional dependency helpers for the execution layer."""

from __future__ import annotations

from typing import Any

from trading_system.execution.exceptions import ExecutionDependencyError


def require_ib_insync() -> Any:
    """Return the `ib_insync` module or raise a dependency error."""

    try:
        import ib_insync
    except ModuleNotFoundError as exc:
        raise ExecutionDependencyError(
            "ib_insync is required for the IBKR execution layer. Install the 'execution' extra."
        ) from exc

    return ib_insync
