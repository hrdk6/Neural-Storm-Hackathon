import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from clinictune import imaging, quality, synthetic
from clinictune.data.loaders import load_shenzhen, load_tbx11k, referable_dr
from clinictune.data.manifest import read_manifest, write_manifest
from clinictune.licences import load_registry, unverified
from clinictune.site.ingest import PackError, read_pack


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_shenzhen_loader_and_manifest_roundtrip(tmp_path):
    for name in ("CHNCXR_0001_0.png", "CHNCXR_0002_1.png", "readme.png"):
        Image.new("L", (32, 32)).save(tmp_path / name)
    recs = load_shenzhen(tmp_path)
    assert [r.label for r in recs] == [0, 1]
    assert recs[1].patient_id == "shenzhen:0002"
    write_manifest(recs, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == recs


def test_tbx11k_loader(tmp_path):
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "TBX11K_train.txt").write_text("tb/tb0001.png\nhealth/h0001.png\n")
    (tmp_path / "lists" / "TBX11K_val.txt").write_text("sick/s0001.png\n")
    recs = load_tbx11k(tmp_path)
    assert [(r.label, r.official_split) for r in recs] == [(1, "train"), (0, "train"), (0, "val")]


def test_referable_dr():
    assert [referable_dr(g) for g in ("0", "1", "2", "4", "")] == [0, 0, 1, 1, None]


def test_registry_lists_every_loader_dataset():
    reg = load_registry()
    for name in ("tbx11k", "shenzhen", "montgomery", "mendeley_tb", "aptos2019", "idrid", "messidor2"):
        assert "licence" in reg["datasets"][name]
    assert unverified(["demo-randproj"]) == []
    with pytest.raises(KeyError):
        unverified(["not-a-dataset"])


def test_gate_accepts_synthetic_and_rejects_wrong_modality():
    rng = np.random.default_rng(0)
    p_tb = synthetic.TRAIN_SITES[0]
    p_dr = synthetic.TRAIN_SITES[3]
    cxr = synthetic.cxr(1, p_tb, rng)
    fundus = synthetic.fundus(1, p_dr, rng)
    assert quality.check("tb", cxr).passed
    assert quality.check("dr", fundus).passed
    assert not quality.check("tb", fundus).passed  # colour fundus is not a chest X-ray
    assert not quality.check("dr", cxr).passed
    flat = np.full((512, 512, 3), 128, np.uint8)
    assert not quality.check("tb", flat).passed
    small = cxr[::4, ::4]
    assert "resolution" in [c.name for c in quality.check("tb", small).failed]


def test_decode_rejects_garbage():
    with pytest.raises(imaging.ImageDecodeError):
        imaging.decode(b"not an image")


def test_read_pack_validation():
    img = _png(np.zeros((64, 64, 3), np.uint8))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil/a.png", img)  # traversal names are reduced to their basename, never extracted
        zf.writestr("labels.csv", "filename,label\na.png,1\n")
    items = read_pack(buf.getvalue(), 100, 10_000_000)
    assert items[0][0] == "a.png" and items[0][2] == 1

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.png", img)
    with pytest.raises(PackError, match="labels.csv"):
        read_pack(buf.getvalue(), 100, 10_000_000)
    with pytest.raises(PackError, match="too large"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("labels.csv", "filename,label\n" + "x" * 2000)
        read_pack(buf.getvalue(), 100, 1000)
