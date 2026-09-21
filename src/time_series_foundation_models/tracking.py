"""Lightweight local experiment registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    created_at_utc: str
    experiment: str
    model: str
    metrics: dict[str, float]
    params: dict[str, Any]
    notes: str = ""


def create_run(
    experiment: str,
    model: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    notes: str = "",
) -> ExperimentRun:
    """Create one immutable run record."""
    return ExperimentRun(
        run_id=str(uuid4()),
        created_at_utc=datetime.now(UTC).isoformat(),
        experiment=experiment,
        model=model,
        metrics=metrics,
        params=params,
        notes=notes,
    )


def append_run(run: ExperimentRun, path: Path) -> Path:
    """Append a run to a JSONL registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(run), sort_keys=True) + "\n")
    return path


def read_runs(path: Path) -> list[ExperimentRun]:
    """Load all experiment runs."""
    if not path.exists():
        return []

    runs: list[ExperimentRun] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                runs.append(ExperimentRun(**json.loads(line)))
    return runs
