"""Reproducible downloads of the public datasets ClinicTune is trained and evaluated on.

``clinictune fetch <name> --root data/raw`` puts each dataset in its own directory (``data/raw/<name>``),
resumes interrupted downloads, and writes a ``labels.csv`` where the source has no simple label file.
Nothing here is redistributed by this repository: every file comes from the source listed in
``datasets.yaml`` under that source's own terms.

Downloaded files are treated as untrusted data: they are only ever *read* as images / CSV / parquet, never
executed or imported.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

NLM = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
HF = "https://huggingface.co/datasets"
UA = {"User-Agent": "clinictune-fetch/0.2 (+research; screening-aid)"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)  # noqa: S310 - fixed https sources only
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def download(url: str, dest: Path, expected_size: int | None = None, retries: int = 4) -> Path:
    """Download to ``dest`` via a ``.part`` file; skip if it already exists with the expected size."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)  # noqa: S310
            with urllib.request.urlopen(req, timeout=300) as r, part.open("wb") as fh:  # noqa: S310
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise OSError(f"size mismatch for {url}: {part.stat().st_size} != {expected_size}")
            part.replace(dest)
            return dest
        except OSError as exc:
            log.warning("download failed (%s/%s) %s: %s", attempt + 1, retries, url, exc)
    raise OSError(f"giving up on {url}")


def _parallel(jobs: Iterable[tuple[str, Path, int | None]], workers: int = 8) -> int:
    jobs = list(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, u, d, s) for u, d, s in jobs]
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                log.info("downloaded %d/%d", done, len(jobs))
    return done


# ----------------------------------------------------------------------------- NLM directory listings

_ROW = re.compile(r"<a href=['\"]([^'\"?]+)['\"]>[^<]*</a></td><td class=['\"]size['\"]>([^<]+)</td>", re.IGNORECASE)


def _nlm_listing(url: str) -> list[tuple[str, int | None]]:
    """Parse an NLM 'Index of' page -> [(name, size or None for directories)]."""
    html = _get(url + "index.html").decode("utf-8", "replace")
    out = []
    for name, size in _ROW.findall(html):
        if name.startswith(("..", "/", "http")) or name == "index.html":
            continue
        size = size.strip()
        out.append((name, None if not size.isdigit() else int(size)))
    return out


def fetch_nlm(base: str, subdirs: list[str], out: Path, workers: int = 8) -> int:
    jobs = []
    for sub in subdirs:
        url = f"{base}/{sub}/"
        for name, size in _nlm_listing(url):
            if size is None:  # nested directory
                continue
            jobs.append((url + name, out / sub / name, size))
    log.info("%s: %d files", base.rsplit("/", 1)[-1], len(jobs))
    return _parallel(jobs, workers)


def fetch_shenzhen(out: Path) -> None:
    fetch_nlm(f"{NLM}/Shenzhen-Hospital-CXR-Set", ["CXR_png", "ClinicalReadings"], out)


def fetch_montgomery(out: Path) -> None:
    fetch_nlm(
        f"{NLM}/Montgomery-County-CXR-Set/MontgomerySet",
        ["CXR_png", "ClinicalReadings", "ManualMask/leftMask", "ManualMask/rightMask"],
        out,
    )


# ----------------------------------------------------------------------------- Google Drive (TBX11K)


def fetch_tbx11k(out: Path) -> None:
    file_id = "1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR"  # official link from github.com/yun-liu/tuberculosis
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    zpath = download(url, out / "TBX11K.zip", expected_size=3306757175)
    if not (out / "TBX11K").exists():
        log.info("extracting %s", zpath)
        _safe_extract(zpath, out)


def _safe_extract(zpath: Path, out: Path) -> None:
    root = out.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            target = (out / info.filename).resolve()
            if not target.is_relative_to(root):
                raise OSError(f"unsafe path in archive: {info.filename}")
        zf.extractall(out)  # noqa: S202 - every member path validated above


# ----------------------------------------------------------------------------- Hugging Face parquet mirrors


def _hf_parquet_to_images(repo: str, out: Path, label_of: Callable[[dict], int], splits: dict[str, str]) -> None:
    """Write each parquet row's image bytes to ``images/`` and a labels.csv (filename, grade, official_split)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    rows = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    for f in sorted(files):
        split = next((v for k, v in splits.items() if k in Path(f).name), "train")
        local = hf_hub_download(repo, f, repo_type="dataset", cache_dir=str(out / ".hf_cache"))
        table = pq.read_table(local)
        for i, rec in enumerate(table.to_pylist()):
            img = rec["image"]
            data = img["bytes"]
            ext = (Path(img.get("path") or "").suffix or ".png").lower()
            name = f"{split}_{Path(f).stem.split('-')[1] if '-' in Path(f).stem else 0}_{i:05d}{ext}"
            (out / "images" / name).write_bytes(data)
            rows.append((name, label_of(rec), split))
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "grade", "official_split"])
        w.writerows(rows)
    log.info("%s: %d images", repo, len(rows))


def fetch_idrid(out: Path) -> None:
    _hf_parquet_to_images(
        "amin-nejad/idrid-disease-grading", out, lambda r: int(r["label"]), {"train": "train", "test": "test"}
    )


def fetch_aptos(out: Path) -> None:
    _hf_parquet_to_images("bumbledeep/aptos", out, lambda r: int(r["label_code"]), {"train": "train"})


# ----------------------------------------------------------------------------- DDR (via the FLARE mirror)

DDR_REPO = "YukunZhou/MICCAI_FLARE_diabetic_retinopathy"
DDR_PER_GRADE = {0: 800, 1: 150, 2: 700, 3: 150, 4: 300}
DDR_UNGRADABLE = 400


def fetch_ddr(out: Path, seed: int = 0) -> None:
    """A stratified subset of DDR (CC BY 4.0): graded images per ICDR grade plus ungradable images,
    whose labels train ClinicTune's learned gradability gate."""
    import random

    from huggingface_hub import HfApi

    out.mkdir(parents=True, exist_ok=True)
    paths = {
        Path(f.path).name: (f.path, f.size)
        for f in HfApi().list_repo_tree(DDR_REPO, repo_type="dataset", recursive=True)
        if hasattr(f, "size") and f.path.endswith(".jpg")
    }

    def read_csv(name: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(_get(f"{HF}/{DDR_REPO}/resolve/main/{name}").decode())))

    grades = {
        r["Name"]: int(r["Label"])
        for s in ("train", "test")
        for r in read_csv(f"MICCAI_FLARE_diabetic_retinopathy_{s}.csv")
    }
    ungradable = [
        r["Name"]
        for s in ("train", "test")
        for r in read_csv(f"MICCAI_FLARE_image_quality_{s}.csv")
        if r["Label"] == "1"
    ]
    rng = random.Random(seed)
    chosen: list[tuple[str, int, int]] = []  # name, grade (-1 = ungradable), gradable flag
    for g, n in DDR_PER_GRADE.items():
        names = sorted(k for k, v in grades.items() if v == g and k in paths)
        chosen += [(k, g, 1) for k in rng.sample(names, min(n, len(names)))]
    ung = sorted(k for k in ungradable if k in paths and k not in grades)
    chosen += [(k, -1, 0) for k in rng.sample(ung, min(DDR_UNGRADABLE, len(ung)))]

    jobs = [(f"{HF}/{DDR_REPO}/resolve/main/{paths[k][0]}", out / "images" / k, paths[k][1]) for k, _, _ in chosen]
    _parallel(jobs, workers=8)
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "grade", "gradable", "official_split"])
        w.writerows((k, g, q, "flare") for k, g, q in chosen)
    log.info("DDR subset: %d images", len(chosen))


FETCHERS: dict[str, Callable[[Path], None]] = {
    "shenzhen": fetch_shenzhen,
    "montgomery": fetch_montgomery,
    "tbx11k": fetch_tbx11k,
    "idrid": fetch_idrid,
    "aptos2019": fetch_aptos,
    "ddr": fetch_ddr,
}
