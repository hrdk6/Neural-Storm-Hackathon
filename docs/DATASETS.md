# Datasets, weights and their terms

The machine-readable source of truth is
[`src/clinictune/data/datasets.yaml`](../src/clinictune/data/datasets.yaml). `clinictune licences` prints
it; `clinictune fetch` downloads from exactly the sources listed there; `clinictune train` refuses to run
while any dataset in the manifest is unverified unless `--allow-unverified-licences` is passed for a local
research run.

**Status.** Every source page below was read on 2026-11-03 while building the registry (`source_checked_on`).
`verified` stays `false` until a person on the team signs the terms off for their own use. Terms change, and
some sources (TBX11K, APTOS) restrict use to research, so check again before publishing or deploying.

## Used in the shipped bundle and the validation results

| Hospital / network | Task | Images used | Fetched from | Terms | Patient IDs |
|---|---|---|---|---|---|
| Shenzhen No.3 People's Hospital, China | TB | 662 (336 TB) | NLM public directory | NLM research release, cite Jaeger 2014 | Yes; sex and age from clinical readings |
| Montgomery County, Maryland, USA | TB | 138 (58 TB) | NLM public directory | NLM research release, cite Jaeger 2014 | Yes; sex, age, manual lung masks |
| TBX11K hospitals, China | TB | 2,200 (800 TB, 700 healthy, 700 sick but not TB) | Official Google Drive link | Research use; no licence file | No (image = patient) |
| DDR, 147 hospitals in 23 Chinese provinces | DR | 2,100 graded + 400 ungradable | Hugging Face FLARE mirror | CC BY 4.0 | No |
| Aravind Eye Hospital, India (APTOS 2019) | DR | 3,662 | Hugging Face mirror (224 px) | Kaggle competition rules (research) | No |
| IDRiD, Nanded, India | DR | 516 | Hugging Face mirror | CC BY 4.0 | No |

Referable diabetic retinopathy is ICDR grade 2 or worse. TB labels are the datasets' own (TBX11K "tb" folder;
NLM file suffix `_1`).

Evaluation splits:
- **TBX11K** test labels are withheld for its online challenge, so only the labelled train and val lists are used.
- **APTOS** only has labels for `train.csv`; all of its images come from that file.
- Every split in the code is by patient (`assert_no_patient_leakage` runs on each one). Where a source releases
  no patient IDs, image = patient, and the same patient could in principle appear on both sides of a split.
- In leave-one-hospital-out experiments the held-out hospital is never used for training, calibration or model
  selection; its adaptation pool and test half are split by patient for every repeat.

## Model weights

| Backbone | Used for | Licence | CPU time per image (Intel Core Ultra 5 laptop) |
|---|---|---|---|
| RAD-DINO (Microsoft), exported at 392 px | TB | MIT; model card says research only | ~450 ms |
| DINOv2-small (Meta) | DR | Apache-2.0 | ~33 ms |
| DINOv2-base (Meta) | compared | Apache-2.0 | ~99 ms |
| MedSigLIP-448 (Google) | supported, not shipped | HAI-DEF terms, gated | needs a Hugging Face login |

## Rules the code enforces

- Data, embeddings, bundles and weights are git-ignored, and CI fails if any are tracked.
- Downloaded archives are extracted with path checks; nothing inside them is ever executed (TBX11K ships code).
- Held-out sites (`--exclude-sites`) never touch training, calibration or model selection.
