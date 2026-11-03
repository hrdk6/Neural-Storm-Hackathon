# Datasets, weights and their terms

The machine-readable source of truth is
[`src/clinictune/data/datasets.yaml`](../src/clinictune/data/datasets.yaml). `clinictune licences` prints
it, and `clinictune train` refuses to run while any dataset in the manifest is `verified: false`.

**Status: every real dataset entry is currently `verified: false`.** The terms below are the project's
best understanding. Re-read each licence at its source, record `verified_on` / `verified_by`, and only
then flip `verified` to true. Terms change.

| Name | Task | Source | Terms (to verify) | Patient IDs | Split used for evaluation |
|---|---|---|---|---|---|
| TBX11K | TB | mmcheng.net/tb | Research use; test labels withheld | No (image = patient) | Official **val** (stated in every report) |
| Shenzhen (NLM) | TB | NLM LHC downloads | NLM research release; cite Jaeger 2014 | Yes | Patient-level split |
| Montgomery (NLM) | TB | NLM LHC downloads | NLM research release; cite Jaeger 2014 | Yes | Patient-level split; lung masks for the shortcut test |
| Mendeley TB | TB | Mendeley Data (record DOI to be fixed) | Usually CC BY 4.0, depending on the record | No | Held-out site |
| APTOS 2019 | DR | Kaggle competition | Competition rules (non-commercial) | No | Carved from train.csv |
| IDRiD | DR | IEEE DataPort | CC BY 4.0 | No | Official train/test |
| Messidor-2 (optional) | DR | ADCIS | Licence agreement, research only | Exam pairs | Patient (exam)-level split |
| MedSigLIP-448 | backbone | Hugging Face (gated) | HAI-DEF Terms of Use | – | – |
| DINOv2-small | backbone | Hugging Face | Apache-2.0 | – | – |

Rules the code enforces:
- Data, embeddings, bundles and weights are git-ignored, and CI fails if any are tracked.
- Splits are by patient (`assert_no_patient_leakage` runs on every split).
- Held-out sites (`--exclude-sites`) are never used for training, calibration or model selection.
