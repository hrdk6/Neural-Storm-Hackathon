# ClinicTune: a site-adaptive screening aid

**Neural Storm 2026 · HI-01 · Medical Image Diagnostic Assistant for under-resourced clinics**

ClinicTune screens **chest X-rays for tuberculosis** (core task) and **colour fundus photos for referable
diabetic retinopathy** (second task). For every image it returns one of three outputs:

| Output | Meaning | What the clinic does |
|---|---|---|
| **Refer** | Signs that need confirmation | Send for confirmatory testing (e.g. sputum NAAT, ophthalmologist) |
| **Low risk** | No signs found *and* the model is confident at this site | Routine care. Symptoms still need to be assessed |
| **Uncertain** | Bad image, unfamiliar image, or the model cannot decide | Retake, or escalate to a senior clinician |

Each output comes with a quality-gate report, a heatmap and a plain-language note in English, Hindi or Kannada.

> **Screening aid / decision support only.** ClinicTune does not diagnose disease, is not a medical
> device, and has not been clinically validated. Every output must be reviewed by a qualified clinician.

---

## Why "site-adaptive"

Published evidence shows that models degrade when moved to a new hospital. TBX11K → Mendeley TB, for
example, reaches about 75% with a frozen backbone and 10 labels per class, and about 91% with 100. Models
can also learn shortcuts such as hospital markers. Conformal guarantees weaken under shift and per
class/subgroup (Mehrtens et al. 2025; Boman et al. 2026). ClinicTune is built around that evidence:

1. **Frozen foundation embeddings + small linear heads.** The backbone is MedSigLIP-448, with DINOv2-small as a
   lightweight CPU fallback. Both run through ONNX Runtime. No fine-tuning, so a clinic laptop can run it.
2. **Site onboarding.** A new clinic uploads 50–100 locally labelled images. ClinicTune then:
   - measures **dataset shift**: a domain-classifier AUC, an MMD permutation test, and a label-shift
     (prevalence) estimate;
   - reports **before** metrics: the training-site calibration applied to local data. This is usually where
     conformal coverage fails, and the report says so;
   - **recalibrates** with MAP Platt scaling, a sensitivity-targeted referral threshold, a **class-conditional
     (Mondrian) conformal layer** and an OOD threshold;
   - reports **after** metrics, **cross-fitted** so that no image is scored by a calibration fitted on it;
   - prints a **site report card** with a verdict (*ready / use with caution / not ready*) and its reasons.
3. **A quality gate first.** It checks resolution, exposure, sharpness and modality. For CXR it also checks
   greyscale, aspect ratio and that both lung fields are visible. For fundus it checks the field of view,
   colour and illumination. An **embedding-space OOD check** is the second line of defence.
4. **Honest reporting.** Every result reports sensitivity, specificity, ROC-AUC, PR-AUC and ECE per site,
   with confidence intervals. Coverage failures are reported, not hidden.

### Decision rule (sensitivity-first)

```
quality gate failed                         -> Uncertain (retake)
embedding out of distribution               -> Uncertain (escalate)
conformal set = {refer}                     -> Refer
set = {refer, low} and p >= site threshold  -> Refer
set = {low}        and p <  site threshold  -> Low risk
anything else (ambiguous / empty / conflict)-> Uncertain (escalate)
```

"Low risk" requires the conformal layer and the threshold to agree. Missing disease is the costlier error.

### Heatmaps that are exact, not post-hoc

The image embedding is the **mean of the backbone's patch tokens**, and the head is linear. The logit
therefore decomposes **exactly** into per-patch contributions:
`logit = mean_t(w · (token_t − μ)/σ) + b`. One forward pass gives a faithful map of *what moved this
model's score*. The UI states that the map explains the score and does not outline disease.

---

## Architecture

```
            ┌────────── one-off, offline ──────────┐        ┌──────────── serving (CPU) ─────────────┐
 datasets → loaders → patient-level splits          │        │ upload → quality gate → ONNX backbone   │
            │ torch → export backbone → ONNX ───────┼──┬────►│ → OOD check → linear head → site Platt  │
            │ ONNX backbone → embeddings (cached)   │  │     │ → class-conditional conformal → decision│
            │ → heads, Platt, threshold, conformal, │  │     │ → exact heatmap → note (en/hi/kn)       │
            │   OOD → model bundle (sha256-pinned)  │──┘     │ SQLite: sites, calibration versions,    │
            └───────────────────────────────────────┘        │ onboarding jobs, audit log, monitoring  │
                                                             └─────────────────────────────────────────┘
```

Embeddings for training are extracted **with the same ONNX model and preprocessing code used in
production**, so there is no train/serve skew.

| Path | What |
|---|---|
| `src/clinictune/data/` | Loaders (TBX11K, Shenzhen, Montgomery, Mendeley TB, APTOS, IDRiD, Messidor-2, generic folder/CSV), manifest, patient-level splits with leakage checks, **licence registry** (`datasets.yaml`) |
| `src/clinictune/embed/` | ONNX backbone contract, MedSigLIP/DINOv2 export, embedding cache, demo backbone |
| `src/clinictune/model/` | Linear heads, calibration, conformal, shift tests, OOD, metrics, bundle format, training |
| `src/clinictune/site/` | Onboarding (shift → before → cross-fitted after → deploy), zip ingestion, report card |
| `src/clinictune/quality.py` | Quality / gradability gate |
| `src/clinictune/triage.py` | Per-image pipeline and decision rule |
| `src/clinictune/experiments.py` | Leave-one-site-out experiments, shortcut tests |
| `src/clinictune/api/` | FastAPI service, SQLite store |
| `web/` | No-build web UI (screen · sites & onboarding · monitoring · about) |

---

## Quick start (synthetic demo, about 2 minutes)

```bash
uv sync
```

```bash
uv run clinictune demo
```

```bash
uv run clinictune serve --port 8000
```

Open http://localhost:8000.

1. **Screen.** Pick *Fundus · DR* and upload any image from `data/onboarding/dr_clinic_new_unseen/` (file
   names start with the true label). Without onboarding, this shifted clinic's images come back
   **Uncertain: out of distribution**. That is intended.
2. **Sites & onboarding.** Create a site, then upload `data/onboarding/dr_clinic_new.zip`. The report card
   appears when the job finishes.
3. **Screen** again with that site selected. The site calibration is now used.

The demo uses **synthetic cartoon images** and a **random-projection backbone**. It exercises every code
path and has **no clinical meaning**. The UI shows a banner saying so.

## Using real data

```bash
uv sync --extra export
```

```bash
uv run clinictune export-backbone dinov2-small --out models/backbones/dinov2-small
```

```bash
uv run clinictune export-backbone medsiglip-448 --out models/backbones/medsiglip-448
```

MedSigLIP is gated. Accept the Health AI Developer Foundations terms on Hugging Face and run `hf auth login`
before exporting it.

```bash
uv run clinictune benchmark models/backbones/dinov2-small models/backbones/medsiglip-448
```

```bash
uv run clinictune manifest shenzhen=/data/ChinaSet_AllFiles/CXR_png montgomery=/data/MontgomerySet/CXR_png tbx11k=/data/TBX11K mendeley_tb=/data/mendeley_tb aptos2019=/data/aptos2019 idrid=/data/IDRiD --out data/manifest.csv
```

```bash
uv run clinictune licences --manifest data/manifest.csv --strict
```

```bash
uv run clinictune loso --manifest data/manifest.csv --backbone-dir models/backbones/medsiglip-448 --tasks tb dr --repeats 20 --out reports/real
```

```bash
uv run clinictune train --manifest data/manifest.csv --backbone-dir models/backbones/medsiglip-448 --bundle-dir models/medsiglip-v1 --tasks tb dr --exclude-sites mendeley_tb
```

```bash
uv run clinictune shortcut --manifest data/manifest.csv --bundle-dir models/medsiglip-v1 --task tb
```

`train` refuses to run while any dataset in the manifest has an unverified licence. Read the licence, then
set `verified: true` in `src/clinictune/data/datasets.yaml`. For local research runs you can pass
`--allow-unverified-licences`.

### Onboarding pack format

A zip containing images plus `labels.csv`:

```
filename,label,patient_id
img_001.png,1,P0001
img_002.png,0,P0002
```

`label` is 1 for a confirmed positive (TB by NAAT/culture, or referable DR by grader) and 0 otherwise.
`patient_id` is optional but recommended. Cross-fitting splits by patient, so leaving it out means two
images of the same patient could land on both sides of a split.

---

## Results

### Real datasets: not yet run in this repository

No real-data numbers are reported here yet. The datasets have to be downloaded under their own licences,
and nothing in this repo should be read as a performance claim. Run the `loso` command above to produce
`reports/real/loso.md` with this table per held-out site:

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) |
|---|---|---|---|---|---|---|---|---|---|---|
| tb | mendeley_tb | source_calibration | 0 | | | | | | | |
| tb | mendeley_tb | site_recalibration | 10 / 50 / 100 | | | | | | | |
| tb | mendeley_tb | site_only_head | 10 / 50 / 100 | – | – | | | – | – | – |

Protocol notes for real data:
- **TBX11K test labels are withheld.** ClinicTune uses the official *val* split as TBX11K's evaluation
  data and labels it as such (`official_splits_used` in every bundle).
- **APTOS** only has labels for `train.csv`, so all APTOS splits are carved from it.
- TBX11K, APTOS and Mendeley TB release no patient IDs, so **image ≈ patient** there. That limitation is
  recorded in the registry.
- The held-out test half of each site is **never** used for fitting, threshold choice or model selection.

### Synthetic sanity check (demo backbone): *not evidence about real performance*

From `reports/synthetic/loso.md` (5 repeats, mean ± SD; coverage target 0.90). Abridged:

| task | held-out site | strategy | k | sens | spec | ROC-AUC | ECE | cov(neg) | cov(pos) |
|---|---|---|---|---|---|---|---|---|---|
| tb | tb_site_a | source_calibration | 0 | 1.00 | 0.00 | 0.98 | 0.64 | **0.00 FAILS** | 1.00 |
| tb | tb_site_a | site_recalibration | 10 | 0.79 ± 0.12 | 0.98 ± 0.04 | 0.98 | 0.23 | 1.00 | 1.00 |
| tb | tb_site_a | site_recalibration | 100 | 0.88 ± 0.06 | 0.95 ± 0.03 | 0.98 | 0.06 | 0.91 | 0.89 |
| tb | tb_clinic_new | source_calibration | 0 | 0.00 | 1.00 | 0.79 | 0.26 | 1.00 | **0.00 FAILS** |
| tb | tb_clinic_new | site_recalibration | 50 | 0.97 ± 0.02 | 0.14 ± 0.09 | 0.79 | 0.04 | 0.95 | 0.97 |
| dr | dr_clinic_new | source_calibration | 0 | 0.99 | 0.12 | 0.88 | 0.30 | **0.30 FAILS** | 1.00 |
| dr | dr_clinic_new | site_recalibration | 50 | 0.79 ± 0.14 | 0.75 ± 0.18 | 0.88 | 0.11 | 0.90 | 0.89 |

What the synthetic run shows. These are mechanism checks, not clinical claims:
- **Negative result, as expected.** Under shift, training-site conformal coverage collapses for one class
  (it misses the target at 6 of 7 held-out sites, and drops to 0% for one class at 4). Site recalibration
  restores per-class coverage near the target.
- **Recalibration cannot fix ranking.** At `tb_clinic_new` the ROC-AUC stays at 0.79 whatever k is.
  Coverage is restored by producing many *Uncertain* outputs, and specificity at the sensitivity target is
  poor. The onboarding verdict for that site is **not ready**. In the same run a head trained only on the
  local labels reaches a higher AUC, which motivates the "local head" item in the roadmap.
- With 10 labels the calibration is held near the training-site prior (by design), and sensitivity
  intervals are wide. The report card shows the Clopper–Pearson interval so this is visible.

**Same experiment with DINOv2-small** (`reports/synthetic_dinov2/loso.md`). A real backbone separates the
synthetic cartoons perfectly (ROC-AUC 1.00 at both new clinics), and **training-site conformal coverage
still fails** at both: 0.00 on negatives at `tb_clinic_new`, 0.73 on positives at `dr_clinic_new`.
Perfect ranking does not make the training-site calibration valid at a new site. With 10 site labels,
coverage is back to 1.00 / 1.00, at sensitivity 1.00 and specificity 1.00 on held-out site images.

### Shortcut test (synthetic)

The synthetic `tb_site_c` has a bright corner marker planted that correlates with the label.
`clinictune shortcut` greys out the image centre and checks whether the border alone still predicts:

| site | original | border only | centre crop 80% |
|---|---|---|---|
| tb_site_a | 0.998 | 0.551 | 0.988 |
| tb_site_b | 0.989 | 0.483 | 0.961 |
| **tb_site_c (marker)** | 0.990 | **0.808 ⚠** | 0.977 |
| tb_clinic_new | 0.772 | 0.646 | 0.652 |

Only the site with the planted shortcut is flagged. For real CXRs, add the Montgomery lung masks with the
`mask_fn` hook to get the lungs-removed condition.

### CPU inference (measured on an Intel Core Ultra 5 225H laptop, ONNX Runtime 1.x, fp32)

| backbone | input | ms / image (batch 1, median) | status |
|---|---|---|---|
| DINOv2-small | 224 | 30 | exported and verified against PyTorch in this repo |
| MedSigLIP-448 | 448 | *not measured here* (gated weights) | run `clinictune benchmark` after export |
| demo random projection | 224 | 0.15 | demo only |

Pick the backbone that meets the clinic's latency budget on its own hardware. The serving image needs no
GPU and no PyTorch.

---

## Deploying

```bash
docker compose up -d --build
```

Set `CLINICTUNE_API_KEY` first. Compose refuses to start without it.

- **Image.** `python:3.12-slim`, ONNX Runtime CPU, non-root user, read-only root filesystem, all
  capabilities dropped, and a health check on `/readyz`. The model bundle is **mounted read-only**, never
  baked into the image.
- **Integrity.** The backbone's sha256 is pinned in the bundle manifest. A tampered or partial file refuses
  to load.
- **Privacy.** Images are processed in memory and never written to disk. The audit log stores only the
  image sha256, the decision and the versions used. No patient identifiers are accepted.
- **Versioning and rollback.** Every onboarding creates a new calibration version. Any version can be
  re-activated (`POST /v1/sites/{id}/calibrations/{task}/{v}/activate`). Calibrations are tied to a
  bundle id, so after a model upgrade a site falls back to training-site settings with a warning until it
  is re-onboarded.
- **Monitoring.** `/metrics` exposes Prometheus counters and latency histograms.
  `/v1/sites/{id}/monitoring` reports the decision mix, gate-failure and OOD rates, and observed
  sensitivity/specificity once confirmatory results are entered (`POST /v1/screenings/{id}/outcome`).
- **Uploads are bounded.** There are per-file and per-pack size limits. Zips are read in memory with
  entry-count and uncompressed-size caps, which blocks zip-slip and zip-bomb attacks. Decompression-bomb
  images are refused.
- **Auth.** An optional shared API key (`X-API-Key`). For multi-user or networked deployments, put the
  service behind a TLS reverse proxy with real authentication. Per-user auth is out of scope by design.

Configuration (environment variables, prefix `CLINICTUNE_`): `BUNDLE_DIR`, `DATA_DIR`, `API_KEY`,
`CORS_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_ONBOARD_ZIP_MB`, `MAX_ONBOARD_IMAGES`, `ORT_THREADS`,
`ONBOARD_WORKERS`, `LOG_LEVEL`, `LOG_JSON`, `STORE_AUDIT_LOG`.

## API

| Method | Path | |
|---|---|---|
| GET | `/healthz`, `/readyz`, `/metrics` | liveness, readiness, Prometheus |
| GET | `/v1/info` | bundle, tasks, internal metrics, languages, disclaimer |
| POST | `/v1/screen` | multipart `file`, `task` (tb/dr), optional `site_id`, `lang` (en/hi/kn), `heatmap` |
| POST | `/v1/screenings/{request_id}/outcome` | `{"outcome": 0 or 1}`, the confirmatory result |
| GET/POST | `/v1/sites` | list / create `{site_id, name}` |
| POST | `/v1/sites/{id}/onboard` | multipart `task`, `pack` (zip), returns `job_id` |
| GET | `/v1/jobs/{job_id}` | onboarding progress / result |
| GET | `/v1/sites/{id}/report[.html]?task=&version=` | report card (JSON / printable HTML) |
| POST | `/v1/sites/{id}/calibrations/{task}/{version}/activate` | rollback / roll forward |
| GET | `/v1/sites/{id}/monitoring?task=&days=` | post-deployment monitoring |

OpenAPI docs are at `/docs`.

## Development

```bash
uv run pytest
```

```bash
uv run ruff check src tests
```

The tests cover: class-conditional coverage holding per class; uninformative conformal sets when a class
has too few labels; calibration with a prior; patient-leakage detection; the exact heatmap decomposition;
the decision table; loaders; the quality gate rejecting the wrong modality; zip-pack validation; bundle
hash verification; that held-out sites are never trained on; and the full API flow (screen → onboard twice
→ report → rollback → outcome → monitoring). CI also fails if data or weights are committed.

## Limitations

- **No real-data validation yet**, and no prospective or clinical evaluation. Synthetic results only show
  that the mechanisms work.
- The quality gate's thresholds are heuristic. Tune and validate them on each real dataset. A learned
  gradability head (for example from EyeQ labels, following the NaIA-RD approach) is a planned upgrade for DR.
- The Hindi and Kannada notes are **drafts** that a native-speaking clinician has not reviewed. The API
  returns `translation_reviewed: false` and the UI shows it.
- Conformal guarantees are **marginal over the site's population** and assume local labels are
  exchangeable with future patients. They are not guarantees for subgroups (age, sex, HIV status, device)
  unless you calibrate per subgroup with enough labels.
- With 50–100 local labels, sensitivity intervals are wide (often ±15 points). The report card shows them,
  so read the interval.
- Datasets without patient IDs (TBX11K, APTOS, Mendeley TB) can leak the same patient across splits.
- Recalibration fixes calibration, not discrimination. A site whose AUC is low needs a local head or a
  different backbone. The verdict says "not ready" in that case.

## Roadmap

- "Local head" onboarding option when cross-fitted evidence shows it beats recalibration (the synthetic
  LOSO suggests it can).
- Learned DR gradability head; CXR view classifier (PA/AP/lateral).
- Per-subgroup conformal calibration when subgroup labels exist.
- Lung-mask shortcut condition wired to the Montgomery masks.

## Licences

Code: Apache-2.0. Datasets and model weights keep **their own** licences. See
[`src/clinictune/data/datasets.yaml`](src/clinictune/data/datasets.yaml) and
[docs/DATASETS.md](docs/DATASETS.md). No data or weights are distributed with this repository.
