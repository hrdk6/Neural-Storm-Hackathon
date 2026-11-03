"""Dataset / weight licence registry (``data/datasets.yaml``)."""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml


def load_registry() -> dict[str, dict[str, Any]]:
    text = resources.files("clinictune.data").joinpath("datasets.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def licence_rows(names: list[str] | None = None) -> list[dict[str, Any]]:
    reg = load_registry()
    rows = []
    for kind in ("datasets", "backbones"):
        for name, entry in reg[kind].items():
            if names and name not in names:
                continue
            rows.append({"kind": kind[:-1], "name": name, **entry})
    return rows


def unverified(names: list[str]) -> list[str]:
    reg = load_registry()
    known = {**reg["datasets"], **reg["backbones"]}
    missing = [n for n in names if n not in known]
    if missing:
        raise KeyError(f"not in licence registry: {missing} - add them to datasets.yaml first")
    return [n for n in names if not known[n].get("verified")]
