"""Configuration for a single-network analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class PipelineConfig:
    """User choices that drive the notebook-equivalent pipeline."""

    db_path: Path
    table: str
    date_column: str
    name_column: str
    value_column: str
    filter_column: str | None = None
    filter_value: str | None = None
    transforms: list[str] = field(default_factory=list)
    measure: str = "distance_correlation"
    date_start: date | None = None
    date_end: date | None = None
    independent_threshold: float = 0.33
    title: str = "Distance Correlation Network"
