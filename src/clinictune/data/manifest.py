"""Manifest CSV: the single tabular description of every image the pipeline touches.

Columns: image_path, label, patient_id, site, task, dataset, official_split
``label`` is binary (1 = refer-worthy finding: TB / referable DR). ``patient_id`` is what splits group on.
``official_split`` keeps the dataset's own train/val/test tag so reports can say which one was used.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

TASKS = ("tb", "dr")


@dataclass(frozen=True, slots=True)
class Record:
    image_path: str
    label: int
    patient_id: str
    site: str
    task: str
    dataset: str
    official_split: str = ""

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0/1, got {self.label!r} for {self.image_path}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if not self.patient_id:
            raise ValueError(f"empty patient_id for {self.image_path}")


FIELDNAMES = [f.name for f in fields(Record)]


def write_manifest(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
            n += 1
    return n


def read_manifest(path: str | Path) -> list[Record]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Record(
            image_path=r["image_path"],
            label=int(r["label"]),
            patient_id=r["patient_id"],
            site=r["site"],
            task=r["task"],
            dataset=r["dataset"],
            official_split=r.get("official_split", "") or "",
        )
        for r in rows
    ]


def summarize(records: Sequence[Record]) -> dict[str, dict[str, int]]:
    """Per-site counts: images, patients, positives."""
    out: dict[str, dict[str, int]] = {}
    by_site: dict[str, list[Record]] = {}
    for r in records:
        by_site.setdefault(r.site, []).append(r)
    for site, recs in sorted(by_site.items()):
        labels = Counter(r.label for r in recs)
        out[site] = {
            "images": len(recs),
            "patients": len({r.patient_id for r in recs}),
            "positive": labels.get(1, 0),
            "negative": labels.get(0, 0),
        }
    return out
