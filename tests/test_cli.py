import json
import zipfile

from clinictune.cli import _num, main
from clinictune.data.manifest import read_manifest


def test_evidence_collects_reports(tmp_path):
    run = tmp_path / "tb_run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"tb": {"backbone": "rad-dino@392", "repeats": 2, "ks": [10]}}))
    (run / "loso_tb_summary.csv").write_text(
        "task,held_out_site,strategy,k_labels,repeats,roc_auc,coverage_pos\n"
        "tb,shenzhen,source_calibration,0,2,0.81,0.55\n"
        "tb,shenzhen,site_recalibration,10,2,0.81,0.91\n"
    )
    (run / "subgroups_tb.csv").write_text("task,held_out_site,variable,group,sensitivity\ntb,shenzhen,sex,F,0.9\n")
    short = tmp_path / "shortcut_tb.csv"
    short.write_text("task,site,condition,roc_auc,n\ntb,shenzhen,border_only,0.52,662\n")
    bench = tmp_path / "bench.jsonl"
    bench.write_text('{"backbone": "rad-dino@392", "ms_per_image_median": 452.1}\n')
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    main(
        [
            "evidence",
            "--bundle-dir",
            str(bundle),
            "--loso",
            str(run),
            "--shortcut",
            str(short),
            "--benchmark",
            str(bench),
        ]
    )
    ev = json.loads((bundle / "evidence.json").read_text())
    rows = ev["loso"]["tb"][0]["summary"]
    assert rows[0]["k_labels"] == 0 and rows[0]["roc_auc"] == 0.81
    assert ev["subgroups"]["tb"][0]["group"] == "F"
    assert ev["shortcut"]["tb"][0]["condition"] == "border_only"
    assert ev["backbones"][0]["ms_per_image_median"] == 452.1


def test_num_parsing():
    assert _num("3") == 3 and _num("0.5") == 0.5 and _num("") is None and _num("tb") == "tb"


def test_pack_from_manifest(demo, tmp_path):
    site = next(r.site for r in read_manifest(demo["manifest"]) if r.site.endswith("clinic_new"))
    out = tmp_path / "pack.zip"
    main(
        [
            "pack",
            "--manifest",
            str(demo["manifest"]),
            "--site",
            site,
            "--n",
            "30",
            "--unseen",
            "5",
            "--max-side",
            "256",
            "--out",
            str(out),
        ]
    )
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "labels.csv" in names and sum(n.startswith("images/") for n in names) == 30
        header = zf.read("labels.csv").decode().splitlines()[0]
        assert header == "filename,label,patient_id,sex,age"
    assert len(list((tmp_path / "pack_unseen").iterdir())) == 5
