"""Command line: ``clinictune <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from clinictune import __version__


def _setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def cmd_fetch(a: argparse.Namespace) -> None:
    from clinictune.data.fetch import FETCHERS

    for name in a.names or sorted(FETCHERS):
        out = Path(a.root) / name
        logging.getLogger("clinictune").info("fetching %s -> %s", name, out)
        FETCHERS[name](out)


def cmd_manifest(a: argparse.Namespace) -> None:
    from clinictune.data.loaders import LOADERS
    from clinictune.data.manifest import read_manifest, summarize, write_manifest

    records = []
    if a.append and Path(a.out).exists():
        records = read_manifest(a.out)
    for spec in a.dataset:
        name, _, root = spec.partition("=")
        if name not in LOADERS:
            sys.exit(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
        if name == "tbx11k" and a.tbx11k_negatives:
            records += LOADERS[name](root, max_per_negative_class=a.tbx11k_negatives, seed=a.seed)
        else:
            records += LOADERS[name](root)
    n = write_manifest(records, a.out)
    print(json.dumps(summarize(records), indent=2))
    print(f"wrote {n} records to {a.out}")


def cmd_licences(a: argparse.Namespace) -> None:
    from clinictune.data.manifest import read_manifest
    from clinictune.licences import licence_rows, unverified

    names = a.names or None
    if a.manifest:
        names = sorted({r.dataset for r in read_manifest(a.manifest)} - {"synthetic"})
    for row in licence_rows(names):
        flag = "verified" if row.get("verified") else "UNVERIFIED"
        print(f"[{flag:10}] {row['kind']:8} {row['name']:14} {row['licence']}")
    if a.strict:
        bad = unverified(names or [])
        if bad:
            sys.exit(f"licence not verified for: {bad} (edit src/clinictune/data/datasets.yaml after checking)")


def cmd_export(a: argparse.Namespace) -> None:
    from clinictune.embed.export import export_backbone
    from clinictune.pipeline import write_backbone_spec

    spec = export_backbone(a.backbone, a.out, size=a.size)
    write_backbone_spec(a.out, spec)
    print(json.dumps(spec.to_dict(), indent=2))


def cmd_benchmark(a: argparse.Namespace) -> None:
    from clinictune.pipeline import load_backbone_dir

    for d in a.backbone_dir:
        bb = load_backbone_dir(d, threads=a.threads)
        print(json.dumps(bb.benchmark(n=a.n, batch_size=a.batch_size)))


def _task_map(values: list[str], tasks: list[str], what: str) -> dict[str, str]:
    """``["tb=dir1", "dr=dir2"]`` or ``["dir"]`` (same for every task) -> {task: value}."""
    out: dict[str, str] = {}
    for v in values:
        if "=" in v:
            t, _, p = v.partition("=")
            out[t] = p
        else:
            for t in tasks:
                out.setdefault(t, v)
    missing = [t for t in tasks if t not in out]
    if missing:
        sys.exit(f"no {what} given for task(s) {missing}")
    return out


def cmd_train(a: argparse.Namespace) -> None:
    from clinictune.data.loaders import load_gradability
    from clinictune.data.manifest import read_manifest
    from clinictune.licences import unverified
    from clinictune.pipeline import train_bundle

    datasets = sorted({r.dataset for r in read_manifest(a.manifest)} - {"synthetic"})
    bad = unverified(datasets)
    if bad and not a.allow_unverified_licences:
        sys.exit(
            f"licences not verified for {bad}. Check them, mark verified in datasets.yaml, "
            "or pass --allow-unverified-licences for local research runs."
        )
    grad = {t: load_gradability(p) for t, p in (v.split("=", 1) for v in a.gradability)} if a.gradability else None
    m = train_bundle(
        a.manifest,
        _task_map(a.backbone, a.tasks, "--backbone"),
        a.bundle_dir,
        tasks=a.tasks,
        exclude_sites=a.exclude_sites,
        cache_dir=a.cache_dir,
        alpha=a.alpha,
        seed=a.seed,
        gradability=grad,
    )
    print(json.dumps(m, indent=2))


def cmd_loso(a: argparse.Namespace) -> None:
    from clinictune.data.manifest import read_manifest
    from clinictune.embed.cache import extract
    from clinictune.experiments import loso, markdown_table, summarize_rows, summarize_subgroups, write_csv
    from clinictune.pipeline import load_backbone_dir

    backbones = _task_map(a.backbone, a.tasks, "--backbone")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# Leave-one-site-out results\n",
        "Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of "
        "the held-out site, which is never used for fitting. Mean ± SD over repeats.\n",
    ]
    meta: dict[str, dict] = {}
    for task in a.tasks:
        bb = load_backbone_dir(backbones[task])
        recs = [r for r in read_manifest(a.manifest) if r.task == task]
        X = extract(recs, bb, Path(a.cache_dir) / f"{bb.spec.name.replace('@', '-')}_{task}_all.npz")
        sub: list[dict] = []
        rows = loso(
            X,
            recs,
            task,
            ks=a.ks,
            repeats=a.repeats,
            alpha=a.alpha,
            seed=a.seed,
            adapt=not a.no_adapt,
            subgroup_rows=sub,
        )
        write_csv(rows, out / f"loso_{task}_raw.csv")
        summary = summarize_rows(rows)
        write_csv(summary, out / f"loso_{task}_summary.csv")
        if sub:
            write_csv(summarize_subgroups(sub), out / f"subgroups_{task}.csv")
        meta[task] = {
            "backbone": bb.spec.name,
            "n_images": len(recs),
            "sites": sorted({r.site for r in recs}),
            "repeats": a.repeats,
            "ks": a.ks,
            "alpha": a.alpha,
        }
        md += [f"\n## {task.upper()} ({bb.spec.name}, {len(recs)} images)\n", markdown_table(summary, a.alpha), ""]
    (out / "loso.md").write_text("\n".join(md), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(md))


def cmd_shortcut(a: argparse.Namespace) -> None:
    from clinictune.data.loaders import montgomery_lung_mask
    from clinictune.data.manifest import read_manifest
    from clinictune.experiments import shortcut_test, write_csv
    from clinictune.model.bundle import Bundle

    b = Bundle.load(a.bundle_dir)
    recs = [r for r in read_manifest(a.manifest) if r.task == a.task]
    masks = {}
    if a.task == "tb":
        for r in recs:
            if r.dataset == "montgomery" and (m := montgomery_lung_mask(r.image_path)) is not None:
                masks[r.image_path] = m
    bb = b.backbone_for(a.task)
    original = Path(a.cache_dir) / f"{bb.spec.name.replace('@', '-')}_{a.task}_all.npz"  # shared with loso/train
    rows = shortcut_test(
        recs,
        b.tasks[a.task],
        bb,
        a.cache_dir,
        lung_masks=masks or None,
        max_per_site=a.max_per_site,
        original_cache=original,
    )
    write_csv(rows, Path(a.out) / f"shortcut_{a.task}.csv")
    for r in rows:
        warn = (
            "  <-- border alone is predictive: possible shortcut"
            if (r["condition"] == "border_only" and r["roc_auc"] > 0.65)
            else ""
        )
        print(f"{r['site']:16} {r['condition']:24} ROC-AUC {r['roc_auc']:.3f} (n={r['n']}){warn}")


def cmd_pack(a: argparse.Namespace) -> None:
    """Make an onboarding pack (zip) from one site of a manifest; the rest of that site is copied to an
    ``_unseen`` folder for trying the screening page on images the site calibration never saw."""
    import random
    import shutil

    from clinictune.data.manifest import read_manifest
    from clinictune.pipeline import make_onboarding_zip

    recs = [r for r in read_manifest(a.manifest) if r.site == a.site]
    if not recs:
        sys.exit(f"no records for site {a.site!r}")
    random.Random(a.seed).shuffle(recs)
    pack, rest = recs[: a.n], recs[a.n :]
    out = make_onboarding_zip(pack, a.out, max_side=a.max_side)
    unseen = Path(str(out).removesuffix(".zip") + "_unseen")
    unseen.mkdir(parents=True, exist_ok=True)
    for r in rest[: a.unseen]:
        shutil.copy2(r.image_path, unseen / f"label{r.label}_{Path(r.image_path).name}")
    n_pos = sum(r.label for r in pack)
    print(f"{out}: {len(pack)} images ({n_pos} positive); {min(len(rest), a.unseen)} unseen in {unseen}")


def cmd_evidence(a: argparse.Namespace) -> None:
    """Collect validation outputs into <bundle>/evidence.json for the UI's Evidence page."""
    import csv as _csv

    def rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return [{k: _num(v) for k, v in r.items()} for r in _csv.DictReader(fh)]

    ev: dict = {"loso": {}, "subgroups": {}, "shortcut": {}, "backbones": [], "datasets": {}}
    for d in map(Path, a.loso):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        for task, m in meta.items():
            ev["loso"].setdefault(task, []).append({**m, "summary": rows(d / f"loso_{task}_summary.csv")})
            if (d / f"subgroups_{task}.csv").exists():
                ev["subgroups"][task] = rows(d / f"subgroups_{task}.csv")
    for p in map(Path, a.shortcut):
        ev["shortcut"][p.stem.split("_")[-1]] = rows(p)
    for p in map(Path, a.benchmark):
        ev["backbones"] += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.manifest:
        from clinictune.data.manifest import read_manifest, summarize

        recs = read_manifest(a.manifest)
        ev["datasets"] = {t: summarize([r for r in recs if r.task == t]) for t in sorted({r.task for r in recs})}
    out = Path(a.bundle_dir) / "evidence.json"
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def _num(v: str) -> object:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v if v != "" else None
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f


def cmd_demo(a: argparse.Namespace) -> None:
    from clinictune.pipeline import build_demo

    paths = build_demo(a.root, n_scale=a.scale, seed=a.seed)
    for k, v in paths.items():
        print(f"{k:16} {v}")


def cmd_onboard(a: argparse.Namespace) -> None:
    from clinictune.api.store import Store
    from clinictune.config import get_settings
    from clinictune.model.bundle import Bundle
    from clinictune.site.ingest import read_pack, run_onboarding

    s = get_settings()
    b = Bundle.load(a.bundle_dir or s.bundle_dir)
    items = read_pack(Path(a.pack).read_bytes(), s.max_onboard_images, 1 << 34)
    calib, report = run_onboarding(b, a.site_id, a.task, items)
    if a.save:
        st = Store(s.db_path)
        if not st.get_site(a.site_id):
            st.create_site(a.site_id, a.site_id)
        print("saved calibration version", st.add_calibration(calib, report))
    from clinictune.api.store import jsonable

    print(json.dumps(jsonable({k: report[k] for k in ("verdict", "verdict_reasons", "warnings", "shift")}), indent=2))


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "clinictune.api.app:app_factory",
        factory=True,
        host=a.host,
        port=a.port,
        workers=a.workers,
        proxy_headers=True,
        log_config=None,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="clinictune", description="Site-adaptive screening aid (decision support).")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="download public datasets (needs --extra fetch)")
    s.add_argument("names", nargs="*", help="shenzhen montgomery tbx11k idrid aptos2019 ddr (default: all)")
    s.add_argument("--root", default="data/raw")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("manifest", help="build a manifest CSV from downloaded datasets")
    s.add_argument("dataset", nargs="+", help="name=path, e.g. shenzhen=/data/ChinaSet_AllFiles/CXR_png")
    s.add_argument("--out", default="data/manifest.csv")
    s.add_argument("--append", action="store_true")
    s.add_argument(
        "--tbx11k-negatives",
        type=int,
        default=None,
        help="keep every TBX11K TB image but sample this many healthy and this many sick-non-TB images",
    )
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("licences", help="show dataset/backbone licences; --strict fails on unverified")
    s.add_argument("names", nargs="*")
    s.add_argument("--manifest")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(fn=cmd_licences)

    s = sub.add_parser("export-backbone", help="export RAD-DINO / DINOv2 / MedSigLIP to ONNX (needs --extra export)")
    s.add_argument("backbone", choices=["rad-dino", "dinov2-small", "dinov2-base", "medsiglip-448"])
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, help="input resolution override (multiple of the patch size)")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("benchmark", help="CPU latency per image for one or more exported backbones")
    s.add_argument("backbone_dir", nargs="+")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--threads", type=int, default=0)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("train", help="extract embeddings (cached) and train a model bundle")
    s.add_argument("--manifest", required=True)
    s.add_argument("--backbone", nargs="+", required=True, help="tb=models/backbones/rad-dino dr=... (or one dir)")
    s.add_argument("--gradability", nargs="*", default=[], help="task=fetched dataset dir with a gradable column")
    s.add_argument("--bundle-dir", required=True)
    s.add_argument("--tasks", nargs="+", default=["tb"], choices=["tb", "dr"])
    s.add_argument("--exclude-sites", nargs="*", default=[], help="held-out sites, never used for training")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--alpha", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-unverified-licences", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("loso", help="leave-one-site-out adaptation experiments")
    s.add_argument("--manifest", required=True)
    s.add_argument("--backbone", nargs="+", required=True, help="tb=dir dr=dir (or one dir for all tasks)")
    s.add_argument("--no-adapt", action="store_true", help="skip the local residual head strategy")
    s.add_argument("--tasks", nargs="+", default=["tb"], choices=["tb", "dr"])
    s.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100])
    s.add_argument("--repeats", type=int, default=10)
    s.add_argument("--alpha", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_loso)

    s = sub.add_parser("shortcut", help="border-only / center-crop shortcut tests")
    s.add_argument("--manifest", required=True)
    s.add_argument("--bundle-dir", required=True)
    s.add_argument("--task", default="tb", choices=["tb", "dr"])
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_shortcut)

    s = sub.add_parser("pack", help="make an onboarding pack from one site of a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--site", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--unseen", type=int, default=40)
    s.add_argument("--max-side", type=int, default=1024)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("evidence", help="bundle validation results (LOSO, shortcut, subgroups) for the UI")
    s.add_argument("--bundle-dir", required=True)
    s.add_argument("--loso", nargs="*", default=[], help="report dirs written by `clinictune loso`")
    s.add_argument("--shortcut", nargs="*", default=[], help="shortcut_<task>.csv files")
    s.add_argument("--benchmark", nargs="*", default=[], help="JSON-lines files from `clinictune benchmark`")
    s.add_argument("--manifest")
    s.set_defaults(fn=cmd_evidence)

    s = sub.add_parser("demo", help="build synthetic data + demo bundle + onboarding packs")
    s.add_argument("--root", default=".")
    s.add_argument("--scale", type=float, default=1.0, help="dataset size multiplier")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("onboard", help="run site onboarding from a pack on the command line")
    s.add_argument("pack")
    s.add_argument("--site-id", required=True)
    s.add_argument("--task", required=True, choices=["tb", "dr"])
    s.add_argument("--bundle-dir")
    s.add_argument("--save", action="store_true", help="store and activate in the service DB")
    s.set_defaults(fn=cmd_onboard)

    s = sub.add_parser("serve", help="run the API + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    _setup(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
