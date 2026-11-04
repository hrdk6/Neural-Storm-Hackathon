# ClinicTune

**A site-adaptive screening aid for tuberculosis on chest X-rays and diabetic retinopathy on fundus photos,
built for under-resourced clinics.** Neural Storm 2026, HI-01.

ClinicTune reads an image and answers in a way a primary health centre can act on: **Refer**, **Low risk**,
or **Uncertain** (escalate or retake). Each answer comes with the reasons, an exact heatmap, the most similar
confirmed cases, and a plain-language note in English, Hindi or Kannada, printed as a screening slip.

> ClinicTune is a **screening aid and decision-support tool. It does not diagnose disease** and is not a
> medical device. Every output must be reviewed by a qualified clinician and confirmed with the appropriate
> test (sputum NAAT for TB, an ophthalmologist for diabetic retinopathy).

![A chest X-ray on the lightbox and its screening slip](docs/screenshots/screen-result.png)

*An unseen chest X-ray from Montgomery County (U.S. National Library of Medicine), screened with a clinic that
is **not onboarded**. The aid refuses to be confident: the slip says **Uncertain**, because the image looks
unlike what the model was calibrated on. The heatmap sits on the upper lung zones, not on the "LT ERECT"
marker.*

## Contents

1. [The problem](#the-problem)
2. [The idea: do not trust the model until it is checked at your clinic](#the-idea)
3. [A tour of the interface](#a-tour-of-the-interface)
4. [How it works](#how-it-works)
5. [Results on real hospitals](#results-on-real-hospitals)
6. [Architecture](#architecture)
7. [Run it](#run-it)
8. [Limitations and roadmap](#limitations)

---

## The problem

AI screening models are usually reported with a single number measured at the hospitals they were built on.
Deployed at a *different* clinic, with a different X-ray machine, camera and patient mix, they quietly get
worse, and nobody at the clinic can tell.

In our experiments, the confidence guarantee set at the training hospitals **failed at all 6 of 6 held-out
hospitals**. One DR model had a good ROC-AUC of 0.88 and still missed most patients who needed an eye
specialist (sensitivity 0.38). ClinicTune is built around that failure.

## The idea

**ClinicTune does not trust itself at a new clinic until it has been checked there.** A clinic uploads 50 to
100 images whose diagnosis was confirmed locally. ClinicTune then:

1. measures how different the clinic is from the training hospitals;
2. shows how badly the unadapted model would have done there;
3. fits the clinic's own settings, and a small adaptation layer when that clearly helps;
4. prints a report card with an honest verdict, including when the answer is "use with caution".

```mermaid
flowchart LR
    A["Public hospital data<br/>(6 hospitals, China / India / USA)"] --> B["Frozen foundation model<br/>RAD-DINO or DINOv2, ONNX"]
    B --> C["Shared model bundle<br/>head, calibration, conformal,<br/>familiarity check, case library"]
    C --> D{"New clinic"}
    E["50 to 100 locally<br/>confirmed images (zip)"] --> D
    D --> F["Onboarding<br/>shift test, before / after,<br/>two strategies compared"]
    F --> G["Clinic settings<br/>(versioned, roll-back-able)"]
    G --> H["Screening at the clinic<br/>Refer / Low risk / Uncertain"]
    H --> I["Monitoring and alerts<br/>confirmed outcomes feed back"]
    I -.->|"drift or missed case"| D
```

---

## A tour of the interface

The UI is a no-build, offline-capable web app with self-hosted fonts (Latin, Devanagari and Kannada).

### Screen

Drop, choose or paste one image. The lightbox sweeps a scan line while the image is read, then a stamp lands
on the screening slip with the next step, the reasons, the heatmap and the six most similar confirmed cases.
**Print slip** gives a one-page referral for the patient's OPD card.

![The empty Screen tab](docs/screenshots/screen-empty.png)

### Queue

Screen a whole morning's batch. The queue sorts itself so referrals rise to the top, and exports to CSV.

![Today's queue sorted by priority](docs/screenshots/queue.png)

### Clinics and the report card

Register a clinic, onboard it with a zip, watch the five onboarding steps run, then read the report card.
Every onboarding is a new version, and any version can be rolled back to.

![Registered clinics and the onboarding form](docs/screenshots/clinics.png)

![Clinic report card for Montgomery County](docs/screenshots/report-card.png)

*The report card after onboarding Montgomery County with 99 local images. With the training hospitals'
settings specificity was 15%. The cross-fitted local adaptation layer lifts ROC-AUC from 0.66 to 0.93 and
specificity to 77%, with sensitivity at 89.7%. The verdict is still **use with caution**, and it says why.*

### Monitoring

Results per day, referral, uncertain and quality-failure rates, and observed sensitivity and specificity once
confirmatory results are entered. Alerts fire on unfamiliar images (equipment change), quality failures, a
significant shift in referral rate, and any confirmed-positive patient who was called low risk.

![Monitoring tab](docs/screenshots/monitoring.png)

### Validation

The evidence, rendered from the model bundle: per-hospital charts, coverage before and after, every number,
backbone comparison, shortcut tests and subgroups. It includes the project's own negative results.

![Validation tab](docs/screenshots/validation.png)

---

## How it works

### One screening, end to end

```mermaid
sequenceDiagram
    autonumber
    actor C as Clinician
    participant UI as Web UI
    participant API as FastAPI service
    participant Q as Quality gate
    participant M as ONNX backbone
    participant S as Clinic settings
    participant DB as SQLite audit log

    C->>UI: drop image, choose task and clinic
    UI->>API: POST /v1/screen
    API->>Q: rules (blur, exposure, modality, lungs visible)
    Q-->>API: pass or retake (+ learned gradability for fundus)
    API->>M: standardised image
    M-->>API: patch tokens, mean embedding
    API->>S: familiarity check, head + adaptation layer, Platt
    S-->>API: score, 90% confidence set
    API->>API: decision table, exact heatmap, similar cases, note
    API->>DB: fingerprint, decision, versions (no image stored)
    API-->>UI: slip
    UI-->>C: stamp, reasons, heatmap, printable slip
```

### How the decision is made

```mermaid
flowchart TD
    S([Image in]) --> Q{"Quality gate<br/>passed?"}
    Q -- no --> U1["Uncertain: retake"]
    Q -- yes --> F{"Familiar to this<br/>clinic's calibration?"}
    F -- no --> U2["Uncertain: escalate"]
    F -- yes --> C{"90% confidence set"}
    C -- "{refer}" --> R1["Refer"]
    C -- "{refer, low risk}" --> T{"Score at or above<br/>the clinic's line?"}
    T -- yes --> R2["Refer<br/>(sensitivity first)"]
    T -- no --> U3["Uncertain: escalate"]
    C -- "{low risk}" --> L{"Score below<br/>the line?"}
    L -- yes --> LR["Low risk"]
    L -- no --> U4["Uncertain: escalate"]
    C -- "empty or other" --> U5["Uncertain: escalate"]
```

"Low risk" needs the score and the confidence check to agree, because missing disease is the costlier error.

### Onboarding a clinic

```mermaid
flowchart LR
    Z["Zip: images +<br/>labels.csv"] --> I["1. Ingest<br/>(zip-slip and zip-bomb safe)"]
    I --> Qc["2. Quality check"]
    Qc --> Sh["3. Shift test<br/>domain classifier AUC, MMD"]
    Sh --> B["4. Before<br/>training hospitals' settings<br/>on local labels"]
    B --> X["5. Compare, cross-fitted<br/>A: recalibrate only<br/>B: local adaptation layer"]
    X --> W{"B beats A by<br/>at least +0.02 ROC-AUC?"}
    W -- yes --> DB["Deploy B"]
    W -- no --> DA["Deploy A"]
    DB --> V["New settings version<br/>+ report card + verdict"]
    DA --> V
```

Recalibration alone (Platt scaling) is monotone, so it cannot change ROC-AUC. The local adaptation layer can,
and is deployed only when it clearly wins out of fold.

### Heatmaps that are exact

The image embedding is the mean of the backbone's patch tokens, and every layer on top of it (the training
head and the clinic's adaptation layer) is linear. The score therefore splits **exactly** into per-patch
contributions, computed in one forward pass. The heatmap explains this model's score; the UI states that it
does not outline disease.

### Fundus standardisation

```mermaid
flowchart LR
    P["Raw fundus photo"] --> Cr["Crop to retinal field"] --> Ce["Centre in a square"] --> Rs["Resize"] --> Mk["Mask outside the circle"] --> Mo["Model"]
    Mo --> Hm["Heatmap"] --> Bk["Mapped back onto the original photo"]
```

The same step runs in training, onboarding and serving. It exists because we found a camera shortcut (see
below).

---

## Results on real hospitals

All numbers come from **public datasets from 6 hospitals or hospital networks in China, India and the USA**:
- **TB:** 3,000 chest X-rays from three sources.
- **DR:** 6,278 fundus photos from three sources, plus 2,500 expert image-quality labels.

The protocol is **leave one hospital out**: train on the other hospitals, then test on the held-out one. Each
held-out hospital is split by patient into a pool the "clinic" labels from and a test half that is never
used for fitting. Results are the mean of 10 repeats. The full tables, with standard deviations, are in
[`reports/real/`](reports/real). Reproduce everything with [`scripts/reproduce.sh`](scripts/reproduce.sh).

```mermaid
flowchart LR
    subgraph Train["Train on the other hospitals"]
        T1["Hospital A"]
        T2["Hospital B"]
    end
    H["Held-out hospital C<br/>never seen"] --> Sp["Patient-level split"]
    Sp --> Pool["Label pool<br/>50 or 100 labels<br/>= the clinic's onboarding"]
    Sp --> Test["Test half<br/>never used for fitting"]
    Train --> Mdl["Shared model"]
    Mdl --> Ad["Adapt with the pool"]
    Pool --> Ad
    Ad --> Ev["Evaluate on test half<br/>10 repeats"]
    Test --> Ev
```

### Tuberculosis: RAD-DINO chest X-ray encoder, CPU only

| Held-out hospital | ROC-AUC, unseen | 90% guarantee before (neg / pos) | After onboarding (neg / pos) | ROC-AUC with local adaptation | Sensitivity / specificity after |
|---|---|---|---|---|---|
| Shenzhen, China | 0.82 | **31% / 72%** fails | 92% / 90% | **0.94** (100 labels) | 0.90 / 0.81 |
| TBX11K hospitals, China | 0.82 | **48% / 83%** fails | 94% / 96% | **0.97** (100 labels) | **0.95 / 0.82** |
| Montgomery County, USA | 0.73 | **4% / 97%** fails | 92% / 91% | **0.89** (50 labels) | 0.88 / 0.59 |

The **WHO target product profile for a TB triage test** asks for sensitivity of at least 90% and
specificity of at least 70%. After onboarding, the aid meets it at TBX11K with 100 labels (94.5% / 82.4%)
and at Shenzhen with 50 labels (91.8% / 73.8%). At Shenzhen with 100 labels it lands just under on
sensitivity (89.8% / 80.7%), and Montgomery, with only 138 images, does not reach it (87.9% / 59.5%). These
are means of 10 repeats; the report card shows each clinic's 95% intervals.

### Diabetic retinopathy: DINOv2-base, CPU only

| Held-out hospital | ROC-AUC, unseen | 90% guarantee before (neg / pos) | After onboarding (neg / pos) | ROC-AUC with local adaptation | Sensitivity / specificity after |
|---|---|---|---|---|---|
| Aravind Eye Hospital, India (APTOS) | 0.94 | **49%** / 99% fails | 90% / 90% | **0.97** | 0.83 / 0.93 |
| DDR, 147 hospitals, China | 0.88 | 94% / **42%** fails | 91% / 90% | **0.90** | 0.84 / 0.78 |
| IDRiD, Nanded, India | 0.94 | **61%** / 96% fails | 92% / 92% | **0.94** | 0.86 / 0.85 |

### What the experiments show, including what went wrong

- **A model that ranks well can still be dangerous at a new hospital.** At DDR the unadapted DR model has
  ROC-AUC 0.88, but its sensitivity is **0.38**: it would miss most patients who need an eye specialist.
  The training hospitals' 90% confidence guarantee failed at **all six** held-out hospitals, three for TB
  and three for DR. Onboarding with 50 to 100 local labels brought every one back to 90% to 96% coverage.
- **Recalibration fixes calibration, not ranking.** Platt scaling is monotone, so it can't change ROC-AUC.
  ClinicTune's **local adaptation layer** can: TB ROC-AUC went 0.82 → 0.94 at Shenzhen and 0.82 → 0.97 at
  TBX11K. Onboarding deploys it only when it beats recalibration by a clear, cross-fitted margin.
- **Starting from the training hospitals beats starting from scratch when labels are scarce.** With 50 local
  labels, the adaptation layer reached mean ROC-AUC 0.924 (TB) and 0.928 (DR); a model trained only on those
  labels reached 0.909 and 0.905. At 100 labels the gap narrows (TB 0.955 vs 0.958, DR 0.936 vs 0.927).
- **Calibration error drops sharply after onboarding.** Expected calibration error fell from 0.54 to 0.11
  at Montgomery and from 0.26 to 0.04 at DDR.
- **Negative result: the guarantee does not hold equally for women.** At Shenzhen, after recalibration with
  100 labels, coverage for TB-positive women was **82%** (sensitivity 0.80), against 94% (sensitivity 0.93)
  for men. This is the failure Boman et al. (2026) warn about: class-conditional calibration is not
  subgroup-conditional. The Validation page shows it in red, and per-subgroup calibration is on the roadmap.
- **A domain model matters for chest X-rays.** On unseen TB hospitals, mean ROC-AUC was 0.59 for
  DINOv2-small, 0.68 for DINOv2-base and **0.79 for RAD-DINO**. For fundus photos the general-purpose
  DINOv2-base was enough: mean 0.92.
- **It reads the lungs, not the markings on the film.** At Montgomery, which the model never trained on,
  ROC-AUC falls from 0.71 to **0.50 (chance) when both lung fields are blanked** with the hospital's manual
  lung masks, and to 0.52 when only the border is left. Border-only scores at the training hospitals are
  0.39 and 0.63. Trimming the border *raises* Montgomery's ROC-AUC to 0.80, so its markers distract rather
  than help.
- **We found a shortcut in the fundus data, and fixed it.** With the whole retina blanked out, the first DR
  model still separated referable from non-referable photos at APTOS (ROC-AUC 0.77) and IDRiD (0.67). The
  cause: in APTOS, the size, position and shape of the retinal field (which reveal the camera) predict
  referable DR **on their own** with ROC-AUC 0.85, because different cameras saw patients of different
  severity. ClinicTune now standardises every fundus photo before the model sees it: crop to the retinal
  field, centre it in a square, resize, and mask outside the circle. The same step runs in training,
  onboarding and serving, and the heatmap maps back onto the original photo. After the fix, the
  retina-blanked score is **0.63 at APTOS, 0.55 at DDR and 0.49 at IDRiD**, all under our 0.65 flag; APTOS
  keeps a little, plausibly from fields the camera cut off at the top or bottom. Unseen-hospital ROC-AUC fell
  only at APTOS (0.954 → 0.938), where the shortcut lived, and rose at DDR (0.870 → 0.875) and IDRiD
  (0.928 → 0.938). We also corrected the test itself along the way: it first filled the blanked retina with
  the retina's own median colour, which leaks overall colour; it now uses a constant fill.

### Choosing a model that runs on a clinic laptop

Every candidate went through the same leave-one-hospital-out test and was timed on a laptop CPU
(Intel Core Ultra 5 225H, no GPU, ONNX Runtime):

| Image model | Input | CPU time per image (median of 20) | Unseen-hospital ROC-AUC, mean | Used for |
|---|---|---|---|---|
| RAD-DINO (chest X-ray foundation model) | 518 px | 1,170 ms | not run (392 px is 2.3x faster) | |
| **RAD-DINO** | **392 px** | **513 ms** | **0.79** (TB) | **TB** |
| DINOv2-base | 224 px | 93 ms | 0.68 (TB), **0.92 (DR)** | **DR** |
| DINOv2-small | 224 px | 34 ms | 0.59 (TB), 0.91 (DR) | |

A full screening, from decoding a 20-megapixel X-ray through the quality gate and heatmap, takes about
2 seconds on this laptop. The serving image needs no GPU and no PyTorch.

---

## What a clinic gets

| | |
|---|---|
| **Screen** | Drop, choose or paste an image. Rule-based quality checks run first (blur, exposure, wrong modality, both lung fields visible); for fundus photos a **learned gradability model** trained on DDR's expert quality labels runs too: on held-out DDR images it catches 94% of ungradable photos while keeping 95% of gradable ones (ROC-AUC 0.98). Then the image is checked for familiarity, scored with the clinic's calibration, given a 90% class-conditional confidence check, and stamped onto a printable slip with the next step, an exact heatmap, the six most similar confirmed reference cases, and the note in the chosen language. |
| **Queue** | Screen a whole morning's batch. The queue sorts itself so referrals rise to the top, and exports to CSV. |
| **Clinics** | Register a clinic and onboard it with a zip of locally confirmed images. Watch the five onboarding steps run, then read the report card: shift test, before and after with 95% intervals, the two adaptation strategies compared, a verdict and warnings. Every onboarding is a new version, and you can roll back. |
| **Monitoring** | Results per day, referral, uncertain and quality-failure rates, and observed sensitivity and specificity once confirmatory results are entered. **Alerts** fire on unfamiliar images (equipment change), quality failures, a significant shift in referral rate (two-proportion test), and any confirmed-positive patient who was called low risk. The screening log downloads as CSV. |
| **Validation** | The evidence above, rendered from the model bundle: per-hospital charts, coverage before and after, every number, backbone comparison, shortcut tests and subgroups. |

---

## Architecture

```mermaid
flowchart TB
    subgraph Offline["Offline, once (CLI)"]
        direction LR
        f["fetch<br/>public data, resumable"] --> ex["export-backbone<br/>RAD-DINO / DINOv2 to ONNX"]
        ex --> lo["loso<br/>leave-one-hospital-out"]
        lo --> tr["train<br/>heads, calibration, conformal,<br/>OOD, gradability, case library"]
        tr --> sc["shortcut<br/>border / lungs / retina blanked"]
        sc --> ev["evidence + pack<br/>validation tables into bundle"]
    end
    ev --> BND[("Model bundle<br/>sha256-pinned")]
    subgraph Serve["Serving, CPU only"]
        direction LR
        WEB["web/<br/>no-build UI"] --> API["FastAPI<br/>api/app.py"]
        API --> PIPE["pipeline.py<br/>quality, ONNX, OOD, head,<br/>conformal, decision"]
        API --> SITE["site/<br/>onboarding, report card"]
        API --> STORE[("SQLite<br/>clinics, versions,<br/>jobs, audit log")]
        API --> MET["Prometheus /metrics"]
    end
    BND --> PIPE
```

| Path | What |
|---|---|
| `src/clinictune/data/` | Reproducible fetcher, loaders (with sex and age from clinical readings, and lung masks), patient-level splits with leakage checks, the **licence registry** |
| `src/clinictune/embed/` | ONNX backbone contract, exporter for RAD-DINO, DINOv2 and MedSigLIP, embedding cache |
| `src/clinictune/model/` | Linear heads, MAP Platt calibration, class-conditional conformal, **local adaptation layer**, shift tests, OOD, metrics, bundle format |
| `src/clinictune/site/` | Onboarding (shift → before → two cross-fitted strategies → deploy), zip ingestion, report card |
| `src/clinictune/experiments.py` | Leave-one-hospital-out, subgroup analysis, shortcut tests |
| `src/clinictune/api/` | FastAPI service, SQLite store, monitoring and alerts |
| `web/` | No-build UI: lightbox and screening slip, self-hosted Anek fonts (Latin, Devanagari, Kannada), works offline |

Embeddings for training are extracted **with the same ONNX model and preprocessing code used in production**,
so there is no train/serve skew.

### Data model

```mermaid
erDiagram
    SITES ||--o{ CALIBRATIONS : "versions"
    SITES ||--o{ SCREENINGS : "logs"
    SITES ||--o{ JOBS : "onboarding runs"
    SITES {
        string site_id PK
        string name
    }
    CALIBRATIONS {
        string task PK
        int version PK
        string bundle_id
        bool active
        json calibration
        json report
    }
    SCREENINGS {
        string request_id PK
        string image_sha256 "fingerprint only, no image"
        string decision
        float probability
        bool gate_passed
        bool ood_flagged
    }
    JOBS {
        string job_id PK
        string task
        string status
        float progress
    }
```

*Abridged from [`store.py`](src/clinictune/api/store.py).*

---

## Run it

### Real model (about 2 to 3 hours on a CPU, about 12 GB download)

```bash
bash scripts/reproduce.sh
```

```bash
CLINICTUNE_BUNDLE_DIR=models/clinictune-v1 uv run clinictune serve
```

Open http://localhost:8000. To try onboarding, register a clinic and upload `data/onboarding/idrid-dr.zip`
(IDRiD, India, held out from training) or `data/onboarding/montgomery-tb.zip`. Then screen images from the
matching `_unseen` folder, which the clinic's settings never saw. File names start with the true label.

### Five-minute synthetic demo (no downloads)

```bash
uv sync
```

```bash
uv run clinictune demo
```

```bash
uv run clinictune serve
```

The demo uses cartoon images and a random-projection backbone. It exercises every code path, means nothing
clinically, and the UI says so on every page.

### CLI

| Command | What it does |
|---|---|
| `fetch`, `manifest`, `licences` | download public datasets, build manifests, show licences |
| `export-backbone`, `benchmark` | export a backbone to ONNX, time it on CPU |
| `train`, `loso`, `shortcut` | train a bundle, leave-one-hospital-out, shortcut tests |
| `pack`, `evidence` | make an onboarding pack, bundle validation evidence |
| `demo`, `onboard`, `serve` | synthetic demo, onboard from the command line, run API and UI |

### Deploy

```bash
CLINICTUNE_BUNDLE=clinictune-v1 docker compose up -d --build
```

Set `CLINICTUNE_API_KEY` first; Compose refuses to start without it. `CLINICTUNE_BUNDLE` names the folder
under `models/` to mount; it defaults to the synthetic `demo` bundle.

- **Image:** python:3.12-slim with ONNX Runtime on CPU. No PyTorch, runs as a non-root user with a read-only
  filesystem and all capabilities dropped, and has a health check.
- **Bundle:** mounted read-only and pinned by sha256, so tampered weights refuse to load.
- **Privacy:** images are processed in memory and never stored. The audit log keeps only a fingerprint, the
  decision and the versions used.
- **HTTP:** strict Content-Security-Policy, bounded uploads, and zip ingestion that resists zip-slip and
  zip bombs.
- **Operations:** Prometheus metrics; rollback of any clinic's settings; an offline-capable UI.

### API

| Method | Path | |
|---|---|---|
| GET | `/healthz`, `/readyz`, `/metrics` | liveness, readiness, Prometheus |
| GET | `/v1/info`, `/v1/evidence` | bundle, tasks and backbones; validation evidence |
| POST | `/v1/screen` | multipart `file`, `task` (tb or dr), optional `site_id`, `lang` (en, hi, kn) |
| POST | `/v1/screenings/{id}/outcome` | record the confirmatory test result |
| GET, POST | `/v1/sites` | list or register clinics |
| POST | `/v1/sites/{id}/onboard` | onboarding pack (zip) as a background job; poll `/v1/jobs/{job_id}` |
| GET | `/v1/sites/{id}/report[.html]` | report card, any version |
| POST | `/v1/sites/{id}/calibrations/{task}/{version}/activate` | roll back or forward |
| GET | `/v1/sites/{id}/monitoring`, `/v1/sites/{id}/screenings.csv` | monitoring with alerts; screening log |

OpenAPI docs are at `/docs`.

### Development

```bash
uv run pytest
```

```bash
uv run ruff check src tests
```

The tests cover:
- per-class conformal coverage, and sets that never exclude a class when it has too few labels;
- calibration with a prior, and the residual adaptation layer fixing a flipped feature;
- patient-leakage detection and the exact heatmap decomposition;
- the decision table and the quality gate;
- fundus-field standardisation (two framings of one retina give the same model view), and heatmaps and
  boxes mapped back onto the original photo;
- loaders and clinical-reading parsing;
- monitoring alerts and the CSV export;
- the evidence and pack commands;
- the full API: screen, onboard twice, report, roll back, record an outcome, monitor.

CI also syntax-checks the UI and fails if data or weights are committed.

---

## Limitations

- **No prospective or clinical validation.** These are retrospective results on public data.
- **Small held-out hospitals give wide intervals.** Montgomery has 138 images, so its pool holds 69. The
  report card shows 95% intervals: read those, not the point estimate.
- **The guarantee is per class, not per subgroup.** The Shenzhen gap for women shows this matters.
- **TBX11K, APTOS and DDR release no patient IDs.** There, image is treated as patient. APTOS comes from a
  224 px mirror.
- **Hindi and Kannada notes are drafts.** A native-speaking clinician has not reviewed them, and the API and
  UI say so.
- **Quality-gate rules are heuristic.** The learned gradability model covers fundus photos only.
- **Licences need a human sign-off.** Every source is recorded and was read on 2026-11-03, but
  [`datasets.yaml`](src/clinictune/data/datasets.yaml) stays `verified: false` until someone on the team
  signs off. TBX11K and APTOS are research-use only.

## Roadmap

- Per-subgroup conformal calibration once a clinic has enough labels per group.
- A learned CXR view and quality classifier (PA, AP or lateral; rotation; inspiration).
- MedSigLIP and RETFound backbones (the exporter supports them; they need gated access).
- Review of the Hindi and Kannada notes by native-speaking clinicians.

## Licences

Code: Apache-2.0. Fonts: Anek by Ek Type, SIL Open Font License (`web/fonts/OFL.txt`). Datasets and model
weights keep **their own** terms; see [docs/DATASETS.md](docs/DATASETS.md). No data or weights are
distributed with this repository.
