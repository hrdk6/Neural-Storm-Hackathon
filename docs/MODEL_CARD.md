# Model card: ClinicTune bundle `clinictune-v1`

Exact identifiers (bundle id, backbone hashes, training counts) are in `models/clinictune-v1/manifest.json`
and `tasks/<task>.json`. Validation tables are in [`reports/real/`](../reports/real) and inside the bundle as
`evidence.json`, which the UI's Validation page renders.

## Intended use
- **What:** a screening aid and decision support for (a) possible tuberculosis on frontal chest X-rays and
  (b) referable diabetic retinopathy (ICDR grade 2 or worse) on colour fundus photos, in clinics without an
  on-site radiologist or ophthalmologist.
- **Who:** trained health workers and medical officers. Every output is reviewed by a clinician and
  confirmed with the reference test (sputum NAAT for TB; ophthalmology review for DR).
- **Not for:** diagnosis; children (not validated); lateral chest X-rays; ultra-widefield or smartphone
  fundus images unless a clinic onboards with them; any use at a clinic that has not been onboarded or whose
  report card says "not ready".

## Models
| | TB | DR |
|---|---|---|
| Frozen backbone | RAD-DINO (Microsoft, MIT licence; research use per model card), ONNX at 392 px | DINOv2-base (Meta, Apache-2.0), ONNX at 224 px |
| Model view | the X-ray as uploaded | `fundus_field:512`: retinal field cropped, centred in a square, outside of the circle blanked (removes a camera-framing shortcut) |
| Head | L2 logistic regression on mean-pooled patch tokens; C by patient-grouped CV | same |
| Calibration | Platt on held-out patients; per clinic: MAP Platt with a pseudo-observation prior | same |
| Clinic adaptation | Optional residual linear layer, chosen only when cross-fitted ROC-AUC improves by at least 0.02 | same |
| Uncertainty | Class-conditional split conformal (α = 0.1), embedding OOD detector | plus a learned gradability gate (DDR quality labels) |
| Explanation | Exact per-patch contribution heatmap; six nearest confirmed reference cases | same |

## Training data (held-out "new clinics" excluded)
- TB: Shenzhen (662) and TBX11K (2,200: all TB plus 700 healthy and 700 sick-but-not-TB). Montgomery (138)
  is held out for onboarding.
- DR: DDR (2,100 graded) and APTOS (3,662). IDRiD (516) is held out for onboarding. The gradability gate
  uses 2,500 DDR images, 400 of them graded ungradable.
- Licences and sources: [DATASETS.md](DATASETS.md).

## Performance
Read the leave-one-hospital-out tables (README, `reports/real/`), not the internal held-out numbers.
Internal numbers come from patients at the training hospitals and overstate performance at a new clinic.
That is the whole point of this project.

## Known failure modes
- **Dataset shift.** The training hospitals' calibration and conformal guarantee failed at all six held-out
  hospitals. Onboarding is required.
- **Subgroups.** At Shenzhen, positive-class coverage after recalibration was 82% for women against 94% for
  men.
- **Small clinics.** With few local positives the aid answers "Uncertain" more often, and intervals are wide.
- **Shortcuts.** In APTOS, the framing of the retinal field alone predicts referable DR (ROC-AUC 0.85). The
  first DR model partly used it (0.77 with the retina blanked). With the standardised model view, the
  retina-blanked score is 0.63 / 0.55 / 0.49 (APTOS / DDR / IDRiD), under the 0.65 flag, with a little left at
  APTOS. For TB, blanking both lung fields drops ROC-AUC to 0.50 (chance).

## Sign-off
| Role | Name | Date |
|---|---|---|
| Technical lead | | |
| Clinical lead | | |
| Language review (Hindi, Kannada) | | |
| Data licences checked | | |
