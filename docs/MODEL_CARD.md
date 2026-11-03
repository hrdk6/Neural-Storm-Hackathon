# Model card: ClinicTune bundle

Fill one copy of this card for every bundle you deploy. Most fields come straight from
`models/<bundle>/manifest.json` and `tasks/<task>.json`.

## Intended use
- **What:** a screening aid / decision support for TB on frontal chest X-rays, and for referable diabetic
  retinopathy (ICDR grade ≥ 2) on colour fundus photos, in clinics where a specialist is not available on site.
- **Who:** trained health workers and clinicians. Every output is reviewed by a clinician.
- **Not for:** diagnosis, children (unless validated locally), lateral CXRs, ultra-widefield or
  smartphone-adapter fundus images (unless onboarded with such images), or any decision without a
  confirmatory test.

## Model
- Backbone: `manifest.backbone.name` (frozen, ONNX, sha256 pinned).
- Head: L2-regularised logistic regression on mean-pooled patch embeddings, with C chosen by patient-grouped CV.
- Calibration: Platt scaling on held-out patients; site recalibration with a pseudo-observation prior.
- Uncertainty: class-conditional split conformal (α in `tasks/<task>.json`), an embedding OOD detector,
  and the quality gate.

## Data
- Training sites and datasets: `training.sites`, `training.datasets`, `training.official_splits_used`.
- Licences: `docs/DATASETS.md`.

## Performance
- Internal held-out patients (training sites only): `metrics.overall`, `metrics.per_site`.
- External / new-site performance: **only** from the site's onboarding report card and the LOSO
  tables (`reports/*/loso.md`). Do not quote internal numbers as expected performance at a new clinic.

## Known failure modes
- Dataset shift (new scanner, exposure, population). The training-site calibration and conformal coverage
  fail at new sites; onboarding is required.
- Shortcut learning (markers, text, collimation). Run `clinictune shortcut` for every new bundle.
- Low-prevalence sites with few labelled positives get mostly *Uncertain* outputs until more positives
  are labelled.

## Sign-off
| Role | Name | Date |
|---|---|---|
| Technical lead | | |
| Clinical lead | | |
| Language review (hi / kn) | | |
