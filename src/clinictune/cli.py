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

    spec = export_backbone(a.backbone, a.out)
    write_backbone_spec(a.out, spec)
    print(json.dumps(spec.to_dict(), indent=2))


def cmd_benchmark(a: argparse.Namespace) -> None:
    from clinictune.pipeline import load_backbone_dir

    for d in a.backbone_dir:
        bb = load_backbone_dir(d, threads=a.threads)
        print(json.dumps(bb.benchmark(n=a.n, batch_size=a.batch_size)))


def cmd_train(a: argparse.Namespace) -> None:
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
    m = train_bundle(
        a.manifest,
        a.backbone_dir,
        a.bundle_dir,
        tasks=a.tasks,
        exclude_sites=a.exclude_sites,
        cache_dir=a.cache_dir,
        alpha=a.alpha,
        seed=a.seed,
    )
    print(json.dumps(m, indent=2))


def cmd_loso(a: argparse.Namespace) -> None:
    from clinictune.data.manifest import read_manifest
    from clinictune.embed.cache import extract
    from clinictune.experiments import loso, markdown_table, summarize_rows, write_csv
    from clinictune.pipeline import load_backbone_dir

    bb = load_backbone_dir(a.backbone_dir)
    out = Path(a.out)
    md = [
        f"# Leave-one-site-out results ({bb.spec.name})\n",
        "Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of "
        "the held-out site, which is never used for fitting. Mean ± SD over repeats.\n",
    ]
    for task in a.tasks:
        recs = [r for r in read_manifest(a.manifest) if r.task == task]
        X = extract(recs, bb, Path(a.cache_dir) / f"{bb.spec.name}_{task}_all.npz")
        rows = loso(X, recs, task, ks=a.ks, repeats=a.repeats, alpha=a.alpha, seed=a.seed)
        write_csv(rows, out / f"loso_{task}_raw.csv")
        summary = summarize_rows(rows)
        write_csv(summary, out / f"loso_{task}_summary.csv")
        md += [f"\n## {task.upper()}\n", markdown_table(summary, a.alpha), ""]
    (out / "loso.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def cmd_shortcut(a: argparse.Namespace) -> None:
    from clinictune.data.manifest import read_manifest
    from clinictune.experiments import shortcut_test, write_csv
    from clinictune.model.bundle import Bundle

    b = Bundle.load(a.bundle_dir)
    recs = [r for r in read_manifest(a.manifest) if r.task == a.task]
    rows = shortcut_test(recs, b.tasks[a.task], b.backbone, a.cache_dir)
    write_csv(rows, Path(a.out) / f"shortcut_{a.task}.csv")
    for r in rows:
        warn = (
            "  <-- border alone is predictive: possible shortcut"
            if (r["condition"] == "border_only" and r["roc_auc"] > 0.65)
            else ""
        )
        print(f"{r['site']:16} {r['condition']:16} ROC-AUC {r['roc_auc']:.3f} (n={r['n']}){warn}")


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

    s = sub.add_parser("manifest", help="build a manifest CSV from downloaded datasets")
    s.add_argument("dataset", nargs="+", help="name=path, e.g. shenzhen=/data/ChinaSet_AllFiles/CXR_png")
    s.add_argument("--out", default="data/manifest.csv")
    s.add_argument("--append", action="store_true")
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("licences", help="show dataset/backbone licences; --strict fails on unverified")
    s.add_argument("names", nargs="*")
    s.add_argument("--manifest")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(fn=cmd_licences)

    s = sub.add_parser("export-backbone", help="export MedSigLIP / DINOv2 to ONNX (needs --extra export)")
    s.add_argument("backbone", choices=["medsiglip-448", "dinov2-small"])
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("benchmark", help="CPU latency per image for one or more exported backbones")
    s.add_argument("backbone_dir", nargs="+")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--threads", type=int, default=0)
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("train", help="extract embeddings (cached) and train a model bundle")
    s.add_argument("--manifest", required=True)
    s.add_argument("--backbone-dir", required=True)
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
    s.add_argument("--backbone-dir", required=True)
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
