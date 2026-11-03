#!/usr/bin/env bash
# Reproduce every real-data number in the README on a CPU-only machine (about 2-3 hours, ~12 GB download).
# Each step is resumable: downloads skip finished files and embeddings are cached in ./cache.
set -euo pipefail

uv sync --extra export --extra fetch --extra dicom

# 1. Public data, each source in its own folder under data/raw (see src/clinictune/data/datasets.yaml)
uv run clinictune fetch shenzhen montgomery tbx11k ddr aptos2019 idrid --root data/raw

# 2. Frozen backbones, exported once to ONNX (verified against PyTorch on export)
uv run clinictune export-backbone rad-dino --size 392 --out models/backbones/rad-dino-392
uv run clinictune export-backbone dinov2-small --out models/backbones/dinov2-small
uv run clinictune export-backbone dinov2-base --out models/backbones/dinov2-base
mkdir -p reports/real
uv run clinictune export-backbone rad-dino --out models/backbones/rad-dino
uv run clinictune benchmark models/backbones/rad-dino models/backbones/rad-dino-392 models/backbones/dinov2-base models/backbones/dinov2-small \
  --n 10 > reports/real/benchmarks.jsonl

# 3. Manifests (patient-level metadata; TBX11K: every TB image + 700 healthy + 700 sick-but-not-TB)
uv run clinictune manifest shenzhen=data/raw/shenzhen montgomery=data/raw/montgomery tbx11k=data/raw/tbx11k \
  --tbx11k-negatives 700 --out data/manifest_tb.csv
uv run clinictune manifest ddr=data/raw/ddr aptos2019=data/raw/aptos2019 idrid=data/raw/idrid --out data/manifest_dr.csv

# 4. Leave-one-hospital-out experiments for every candidate backbone
for b in dinov2-small dinov2-base rad-dino-392; do
  uv run clinictune loso --manifest data/manifest_tb.csv --backbone "models/backbones/$b" --tasks tb \
    --repeats 10 --out "reports/real/tb_$b"
done
for b in dinov2-small dinov2-base; do
  uv run clinictune loso --manifest data/manifest_dr.csv --backbone "models/backbones/$b" --tasks dr \
    --repeats 10 --out "reports/real/dr_$b"
done

# 5. Deployable bundle. Montgomery (USA) and IDRiD (India) are held out as "new clinics" for the onboarding demo.
cat data/manifest_tb.csv > data/manifest.csv && tail -n +2 data/manifest_dr.csv >> data/manifest.csv
uv run clinictune train --manifest data/manifest.csv \
  --backbone tb=models/backbones/rad-dino-392 dr=models/backbones/dinov2-base \
  --gradability dr=data/raw/ddr --tasks tb dr --exclude-sites montgomery idrid \
  --bundle-dir models/clinictune-v1 --allow-unverified-licences

# 6. Shortcut tests with the deployed bundle (lungs blanked with Montgomery's manual masks)
uv run clinictune shortcut --manifest data/manifest_tb.csv --bundle-dir models/clinictune-v1 --task tb --out reports/real
uv run clinictune shortcut --manifest data/manifest_dr.csv --bundle-dir models/clinictune-v1 --task dr --out reports/real

# 7. Evidence for the UI's Validation page
uv run clinictune evidence --bundle-dir models/clinictune-v1 --manifest data/manifest.csv \
  --loso reports/real/tb_dinov2-small reports/real/tb_dinov2-base reports/real/tb_rad-dino-392 \
         reports/real/dr_dinov2-small reports/real/dr_dinov2-base \
  --shortcut reports/real/shortcut_tb.csv reports/real/shortcut_dr.csv --benchmark reports/real/benchmarks.jsonl

# 8. Onboarding packs for the held-out "new clinics" (100 labelled images each; the rest stay unseen)
uv run clinictune pack --manifest data/manifest_tb.csv --site montgomery --n 100 --unseen 38 \
  --out data/onboarding/montgomery-tb.zip
uv run clinictune pack --manifest data/manifest_dr.csv --site idrid --n 120 --unseen 40 \
  --out data/onboarding/idrid-dr.zip

echo "Done. Serve with: CLINICTUNE_BUNDLE_DIR=models/clinictune-v1 uv run clinictune serve"
