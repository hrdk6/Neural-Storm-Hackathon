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


def test_clinical_reading_parser():
    from clinictune.data.loaders import parse_clinical_reading

    assert parse_clinical_reading("Patient's Sex: F \nPatient's Age: 027Y\nnormal") == ("F", "27")
    assert parse_clinical_reading("male 35yrs\nnormal") == ("M", "35")
    assert parse_clinical_reading("female 8 month\nPTB") == ("F", "0")
    assert parse_clinical_reading("nothing useful") == ("", "")


def test_graded_loader_and_gradability(tmp_path):
    from clinictune.data.loaders import load_gradability, load_graded

    (tmp_path / "images").mkdir()
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (32, 32)).save(tmp_path / "images" / n)
    (tmp_path / "labels.csv").write_text(
        "filename,grade,gradable,official_split\na.jpg,0,1,x\nb.jpg,3,1,x\nc.jpg,-1,0,x\n"
    )
    recs = load_graded(tmp_path, dataset="ddr")
    assert [(r.label, r.task) for r in recs] == [(0, "dr"), (1, "dr")]
    assert [y for _, y in load_gradability(tmp_path)] == [0, 0, 1]


def test_montgomery_mask_lookup(tmp_path):
    from clinictune.data.loaders import montgomery_lung_mask

    for sub in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
        (tmp_path / sub).mkdir(parents=True)
        Image.new("L", (8, 8)).save(tmp_path / sub / "MCUCXR_0001_0.png")
    assert montgomery_lung_mask(tmp_path / "CXR_png" / "MCUCXR_0001_0.png") is not None
    assert montgomery_lung_mask(tmp_path / "CXR_png" / "MCUCXR_0002_0.png") is None


def test_standardize_fundus_removes_framing_differences():
    """Two cameras framing the same retina differently must give the same model view."""
    rng = np.random.default_rng(1)
    retina = synthetic.fundus(1, synthetic.TRAIN_SITES[3], rng)  # 512 x 512, field in the centre
    wide = np.zeros((600, 1000, 3), np.uint8)
    wide[40:552, 300:812] = retina  # same retina, letterboxed by a different camera
    a, region_a = imaging.standardize_fundus(retina)
    b, region_b = imaging.standardize_fundus(wide)
    assert a.shape == b.shape and a.shape[0] == a.shape[1]
    small = lambda x: np.asarray(Image.fromarray(x).resize((64, 64))).astype(float)  # noqa: E731
    assert np.abs(small(a) - small(b)).mean() < 6  # nearly identical after standardisation
    assert region_b[0] > region_a[0] and abs(region_b[2] - region_a[2]) <= 3  # offset differs, size does not
    corner = a[: a.shape[0] // 10, : a.shape[1] // 10]
    assert corner.max() == 0  # outside the inscribed circle is blanked


def test_heatmap_maps_back_to_original():
    from clinictune.explain import overlay_png, regions_to_original, top_regions

    rgb = np.zeros((300, 450, 3), np.uint8)
    rgb[20:280, 95:355] = 120
    view, region = imaging.model_view("fundus_field:512", rgb)
    grid = np.zeros((16, 16))
    grid[8, 8] = 1.0
    png = overlay_png(rgb, grid, region=region)
    assert Image.open(io.BytesIO(png)).size[0] == 512  # composite keeps the original's aspect
    (box,) = regions_to_original(top_regions(grid, k=1), region, rgb.shape)
    assert abs(box["x"] - 0.5) < 0.02 and abs(box["y"] - 0.5) < 0.04
    assert imaging.model_view("", rgb)[1] is None
