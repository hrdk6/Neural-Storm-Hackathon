"""End-to-end: synthetic demo bundle -> onboarding -> API screening, reports, monitoring, rollback."""

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clinictune.config import Settings
from clinictune.data.manifest import read_manifest
from clinictune.embed.backbone import IntegrityError, OnnxBackbone
from clinictune.experiments import loso, summarize_rows
from clinictune.model.bundle import Bundle
from clinictune.site.ingest import read_pack, run_onboarding


def test_bundle_loads_and_verifies_hash(demo, tmp_path):
    b = Bundle.load(demo["bundle"])
    assert set(b.tasks) == {"tb", "dr"} and b.is_demo
    emb, tok = b.backbone.embed([np.zeros((300, 300, 3), np.uint8)])
    gh, gw = b.backbone.spec.grid
    assert tok.shape == (1, gh * gw, b.backbone.spec.dim) and emb.shape == (1, b.backbone.spec.dim)
    # Tampered weights must refuse to load.
    bad = tmp_path / "backbone.onnx"
    bad.write_bytes(Path(demo["bundle"], "backbone.onnx").read_bytes() + b"x")
    with pytest.raises(IntegrityError):
        OnnxBackbone(tmp_path, b.backbone.spec)


def test_training_never_sees_held_out_sites(demo):
    b = Bundle.load(demo["bundle"])
    for tm in b.tasks.values():
        assert not any("clinic_new" in s for s in tm.training["sites"])


def test_onboarding_report_reports_before_and_after(demo):
    b = Bundle.load(demo["bundle"])
    items = read_pack(Path(demo["dr_clinic_new"]).read_bytes(), 1000, 1 << 32)
    calib, report = run_onboarding(b, "clinic", "dr", items)
    assert report["verdict"] in {"ready", "use_with_caution", "not_ready"}
    assert report["after_recalibration_crossfit"]["available"]
    assert "class_1" in report["before_recalibration"]["conformal_coverage"]
    assert calib.bundle_id == b.bundle_id
    assert 0.5 <= report["shift"]["domain_auc"] <= 1.0
    json.dumps(report, allow_nan=True)


def test_loso_runs(demo):
    recs = [r for r in read_manifest(demo["manifest"]) if r.task == "dr"]
    b = Bundle.load(demo["bundle"])
    from clinictune.embed.cache import extract

    X = extract(recs, b.backbone, demo["root"] / "cache" / "loso_test.npz")
    rows = loso(X, recs, "dr", ks=(10, 50), repeats=2)
    strategies = {r["strategy"] for r in rows}
    assert {"source_calibration", "site_recalibration", "site_only_head"} <= strategies
    assert summarize_rows(rows)


def _png_bytes(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture()
def client(demo, tmp_path):
    from clinictune.api.app import create_app

    s = Settings(bundle_dir=demo["bundle"], data_dir=tmp_path, api_key="secret", log_json=False)
    with TestClient(create_app(s)) as c:
        c.headers["X-API-Key"] = "secret"
        yield c


def test_api_auth_and_health(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"ready": True}
    assert client.get("/v1/info", headers={"X-API-Key": "wrong"}).status_code == 401
    info = client.get("/v1/info").json()
    assert info["demo"] is True and set(info["tasks"]) == {"tb", "dr"}
    assert "screening aid" in info["disclaimer"].lower()
    assert client.get("/").status_code == 200  # web UI
    assert "clinictune_screenings_total" in client.get("/metrics").text


def test_api_screen_onboard_report_monitor(client, demo):
    unseen = sorted(Path(demo["root"], "data", "onboarding", "dr_clinic_new_unseen").glob("*.png"))
    files = {"file": ("x.png", _png_bytes(unseen[0]), "image/png")}

    r = client.post("/v1/screen", files=files, data={"task": "dr", "lang": "kn"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["decision"] in {"refer", "low_risk", "uncertain"}
    assert res["calibration"] == "training-sites" and res["note"]["lang"] == "kn"
    assert res["note"]["translation_reviewed"] is False
    assert res["heatmap_png"].startswith("data:image/png;base64,")

    # wrong modality is caught by the gate
    rng = np.random.default_rng(0)
    from clinictune import synthetic

    cxr = synthetic.cxr(0, synthetic.TRAIN_SITES[0], rng)
    buf = io.BytesIO()
    Image.fromarray(cxr).save(buf, format="PNG")
    r = client.post("/v1/screen", files={"file": ("c.png", buf.getvalue(), "image/png")}, data={"task": "dr"})
    assert r.json()["decision"] == "uncertain" and r.json()["reason"] == "quality_gate"

    assert (
        client.post("/v1/screen", files={"file": ("a.png", b"junk", "image/png")}, data={"task": "dr"}).status_code
        == 422
    )
    assert client.post("/v1/screen", files=files, data={"task": "xx"}).status_code == 422

    # site lifecycle
    assert client.post("/v1/sites", json={"site_id": "Bad Id", "name": "x"}).status_code == 422
    assert client.post("/v1/sites", json={"site_id": "phc-1", "name": "PHC One"}).status_code == 201
    assert client.post("/v1/sites", json={"site_id": "phc-1", "name": "PHC One"}).status_code == 409
    pack = Path(demo["dr_clinic_new"]).read_bytes()
    for _ in range(2):  # two versions, to test rollback
        r = client.post(
            "/v1/sites/phc-1/onboard", data={"task": "dr"}, files={"pack": ("p.zip", pack, "application/zip")}
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        for _ in range(240):
            job = client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] in {"done", "failed"}:
                break
            time.sleep(0.5)
        assert job["status"] == "done", job

    site = client.get("/v1/sites/phc-1").json()
    assert [c["version"] for c in site["calibrations"]] == [1, 2]
    assert [c["active"] for c in site["calibrations"]] == [0, 1]
    rep = client.get("/v1/sites/phc-1/report", params={"task": "dr"}).json()
    assert rep["version"] == 2 and "shift" in rep
    html = client.get("/v1/sites/phc-1/report.html", params={"task": "dr", "version": 1}).text
    assert "Site report card" in html and "Screening aid" in html
    assert client.post("/v1/sites/phc-1/calibrations/dr/1/activate").status_code == 200
    assert client.get("/v1/sites/phc-1/report", params={"task": "dr"}).json()["version"] == 1

    r = client.post("/v1/screen", files=files, data={"task": "dr", "site_id": "phc-1"}).json()
    assert r["calibration"] == "site:phc-1:v1"
    assert client.post(f"/v1/screenings/{r['request_id']}/outcome", json={"outcome": 1}).status_code == 200
    mon = client.get("/v1/sites/phc-1/monitoring", params={"task": "dr"}).json()
    assert mon["n"] == 1 and mon["confirmed_outcomes"] == 1


def test_upload_limit(demo, tmp_path):
    from clinictune.api.app import create_app

    s = Settings(bundle_dir=demo["bundle"], data_dir=tmp_path, max_upload_mb=0.001, log_json=False)
    with TestClient(create_app(s)) as c:
        r = c.post("/v1/screen", files={"file": ("x.png", b"0" * 5000, "image/png")}, data={"task": "tb"})
        assert r.status_code == 413
