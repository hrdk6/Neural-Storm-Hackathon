"""Dataset loaders. Each turns a dataset as downloaded into a list of :class:`Record`.

Loaders only read file names and label files - they never copy or modify images. Where a dataset has
no patient identifiers the image id is used as the patient id; the registry (datasets.yaml) records
that limitation so reports can repeat it.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from clinictune.data.manifest import Record

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def _images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}")
    return path


# --------------------------------------------------------------------------- TB


def load_tbx11k(root: str | Path, site: str = "tbx11k") -> list[Record]:
    """TBX11K: ``imgs/{health,sick,tb}/*.png`` plus ``lists/TBX11K_{train,val}.txt``.

    ``tb`` -> 1; ``health`` and ``sick`` (non-TB disease) -> 0. The unlabeled ``imgs/test`` folder is skipped.
    """
    root = Path(root)
    lists = _require(root / "lists", "TBX11K lists/ folder")
    out: list[Record] = []
    for split in ("train", "val"):
        list_file = _require(lists / f"TBX11K_{split}.txt", f"TBX11K {split} list")
        for line in list_file.read_text(encoding="utf-8").splitlines():
            rel = line.strip().split()[0] if line.strip() else ""
            if not rel:
                continue
            category = rel.split("/")[0]
            if category not in {"health", "sick", "tb"}:
                continue
            img = root / "imgs" / rel
            out.append(
                Record(
                    image_path=str(img.resolve()),
                    label=int(category == "tb"),
                    patient_id=f"tbx11k:{Path(rel).stem}",
                    site=site,
                    task="tb",
                    dataset="tbx11k",
                    official_split=split,
                )
            )
    return out


_NLM_RE = re.compile(r"^(?P<prefix>CHNCXR|MCUCXR)_(?P<id>\d+)_(?P<label>[01])$", re.IGNORECASE)


def _load_nlm(root: Path, dataset: str, site: str) -> list[Record]:
    out = []
    for p in _images(root):
        m = _NLM_RE.match(p.stem)
        if not m or "mask" in str(p.parent).lower():
            continue
        out.append(
            Record(
                image_path=str(p.resolve()),
                label=int(m["label"]),
                patient_id=f"{dataset}:{m['id']}",
                site=site,
                task="tb",
                dataset=dataset,
            )
        )
    if not out:
        raise FileNotFoundError(f"no {dataset} images matching <PREFIX>_<id>_<0|1>.png under {root}")
    return out


def load_shenzhen(root: str | Path, site: str = "shenzhen") -> list[Record]:
    return _load_nlm(Path(root), "shenzhen", site)


def load_montgomery(root: str | Path, site: str = "montgomery") -> list[Record]:
    return _load_nlm(Path(root), "montgomery", site)


def load_class_folders(
    root: str | Path,
    *,
    task: str,
    site: str,
    dataset: str,
    positive: Iterable[str],
    negative: Iterable[str],
) -> list[Record]:
    """Generic folder-per-class layout (Mendeley TB, or a clinic's own export).

    Folder names are matched case-insensitively against ``positive`` / ``negative``.
    """
    root = Path(root)
    pos = {s.lower() for s in positive}
    neg = {s.lower() for s in negative}
    out = []
    for p in _images(root):
        cls = p.relative_to(root).parts[0].lower()
        if cls in pos:
            label = 1
        elif cls in neg:
            label = 0
        else:
            continue
        out.append(
            Record(
                image_path=str(p.resolve()),
                label=label,
                patient_id=f"{dataset}:{p.stem}",
                site=site,
                task=task,
                dataset=dataset,
            )
        )
    if not out:
        raise FileNotFoundError(f"no images in class folders {sorted(pos | neg)} under {root}")
    return out


def load_mendeley_tb(root: str | Path, site: str = "mendeley_tb") -> list[Record]:
    return load_class_folders(
        root,
        task="tb",
        site=site,
        dataset="mendeley_tb",
        positive=["tuberculosis", "tb", "positive"],
        negative=["normal", "healthy", "negative"],
    )


# --------------------------------------------------------------------------- DR


def load_csv(
    csv_path: str | Path,
    *,
    image_root: str | Path,
    image_col: str,
    label_col: str,
    task: str,
    site: str,
    dataset: str,
    to_binary: Callable[[str], int | None],
    patient_col: str | None = None,
    image_suffix: str = "",
    official_split: str = "",
) -> list[Record]:
    """Generic CSV-of-labels loader. ``to_binary`` returns None to drop a row (e.g. ungradable)."""
    image_root = Path(image_root)
    index = {p.name.lower(): p for p in _images(image_root)}
    index.update({p.stem.lower(): p for p in _images(image_root)})
    out = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            label = to_binary(row[label_col])
            if label is None:
                continue
            key = (row[image_col] + image_suffix).lower()
            img = index.get(key) or index.get(Path(key).stem)
            if img is None:
                continue
            pid = row[patient_col] if patient_col else Path(row[image_col]).stem
            out.append(
                Record(
                    image_path=str(img.resolve()),
                    label=label,
                    patient_id=f"{dataset}:{pid}",
                    site=site,
                    task=task,
                    dataset=dataset,
                    official_split=official_split,
                )
            )
    if not out:
        raise FileNotFoundError(f"no rows of {csv_path} matched images under {image_root}")
    return out


def referable_dr(grade: str) -> int | None:
    """ICDR grade -> referable DR (moderate NPDR or worse). Blank / non-numeric grades are dropped."""
    try:
        g = int(float(grade))
    except ValueError:
        return None
    return int(g >= 2)


def load_aptos(root: str | Path, site: str = "aptos2019") -> list[Record]:
    root = Path(root)
    return load_csv(
        _require(root / "train.csv", "APTOS train.csv"),
        image_root=_require(root / "train_images", "APTOS train_images/"),
        image_col="id_code",
        label_col="diagnosis",
        task="dr",
        site=site,
        dataset="aptos2019",
        to_binary=referable_dr,
        image_suffix=".png",
        official_split="train",
    )


def load_idrid(root: str | Path, site: str = "idrid") -> list[Record]:
    """IDRiD 'B. Disease Grading' release; keeps the official train/test tags."""
    root = Path(root)
    out: list[Record] = []
    for split, needle in (("train", "training labels"), ("test", "testing labels")):
        csvs = [p for p in root.rglob("*.csv") if needle in p.name.lower()]
        if not csvs:
            raise FileNotFoundError(f"IDRiD '{needle}' CSV not found under {root}")
        out += load_csv(
            csvs[0],
            image_root=root,
            image_col="Image name",
            label_col="Retinopathy grade",
            task="dr",
            site=site,
            dataset="idrid",
            to_binary=referable_dr,
            image_suffix=".jpg",
            official_split=split,
        )
    return out


def load_messidor2(
    root: str | Path, labels_csv: str | Path, site: str = "messidor2", exam_col: str | None = None
) -> list[Record]:
    """Messidor-2 with the Krause et al. adjudicated grades (``image_id``, ``adjudicated_dr_grade``,
    ``adjudicated_gradable``). Pass ``exam_col`` if your CSV has an exam/patient column."""

    def grade(v: str) -> int | None:
        return referable_dr(v)

    return load_csv(
        labels_csv,
        image_root=root,
        image_col="image_id",
        label_col="adjudicated_dr_grade",
        task="dr",
        site=site,
        dataset="messidor2",
        to_binary=grade,
        patient_col=exam_col,
    )


LOADERS: dict[str, Callable[..., list[Record]]] = {
    "tbx11k": load_tbx11k,
    "shenzhen": load_shenzhen,
    "montgomery": load_montgomery,
    "mendeley_tb": load_mendeley_tb,
    "aptos2019": load_aptos,
    "idrid": load_idrid,
}
