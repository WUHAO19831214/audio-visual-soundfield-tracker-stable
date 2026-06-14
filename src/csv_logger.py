"""Small CSV helpers used by tracking and fusion workflows."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class CSVLogger:
    """Thread-safe in-memory row collector with explicit CSV export."""

    def __init__(self, fieldnames: Sequence[str]) -> None:
        self.fieldnames = list(fieldnames)
        self._rows: list[dict] = []
        self._lock = threading.Lock()

    def append(self, row: Mapping) -> None:
        normalized = {field: row.get(field, "") for field in self.fieldnames}
        with self._lock:
            self._rows.append(normalized)

    def extend(self, rows: Iterable[Mapping]) -> None:
        with self._lock:
            for row in rows:
                self._rows.append(
                    {field: row.get(field, "") for field in self.fieldnames}
                )

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [row.copy() for row in self._rows]

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def export(self, path: str | Path) -> Path:
        return write_csv(path, self.fieldnames, self.snapshot())


def write_csv(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping]
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path

