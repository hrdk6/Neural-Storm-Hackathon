# Leave-one-site-out results

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## DR (dinov2-base, 6278 images)

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dr | aptos2019 | site_adapt | 50 | 0.86 ± 0.05 | 0.91 ± 0.02 | 0.96 ± 0.01 | 0.94 ± 0.01 | 0.05 ± 0.02 | 0.89 ± 0.06 | 0.93 ± 0.04 | ok |
| dr | aptos2019 | site_adapt | 100 | 0.86 ± 0.04 | 0.92 ± 0.02 | 0.97 ± 0.00 | 0.95 ± 0.01 | 0.04 ± 0.01 | 0.89 ± 0.03 | 0.88 ± 0.03 | ok |
| dr | aptos2019 | site_only_head | 10 | - | - | 0.85 ± 0.06 | 0.77 ± 0.11 | - | - | - |  |
| dr | aptos2019 | site_only_head | 50 | - | - | 0.94 ± 0.01 | 0.91 ± 0.02 | - | - | - |  |
| dr | aptos2019 | site_only_head | 100 | - | - | 0.96 ± 0.01 | 0.93 ± 0.02 | - | - | - |  |
| dr | aptos2019 | site_recalibration | 10 | 0.85 ± 0.15 | 0.87 ± 0.09 | 0.95 ± 0.00 | 0.93 ± 0.00 | 0.08 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | aptos2019 | site_recalibration | 50 | 0.88 ± 0.05 | 0.88 ± 0.04 | 0.95 ± 0.00 | 0.93 ± 0.00 | 0.05 ± 0.02 | 0.89 ± 0.07 | 0.95 ± 0.04 | ok |
| dr | aptos2019 | site_recalibration | 100 | 0.83 ± 0.04 | 0.92 ± 0.02 | 0.95 ± 0.00 | 0.93 ± 0.00 | 0.03 ± 0.01 | 0.90 ± 0.03 | 0.89 ± 0.04 | ok |
| dr | aptos2019 | source_calibration | 0 | 0.99 ± 0.00 | 0.54 ± 0.01 | 0.95 ± 0.00 | 0.93 ± 0.00 | 0.24 ± 0.00 | 0.57 ± 0.01 | 1.00 ± 0.00 | **FAILS** neg |
| dr | ddr | site_adapt | 50 | 0.83 ± 0.08 | 0.69 ± 0.15 | 0.87 ± 0.02 | 0.90 ± 0.02 | 0.06 ± 0.03 | 0.91 ± 0.05 | 0.90 ± 0.05 | ok |
| dr | ddr | site_adapt | 100 | 0.84 ± 0.06 | 0.76 ± 0.09 | 0.89 ± 0.01 | 0.92 ± 0.01 | 0.04 ± 0.01 | 0.90 ± 0.04 | 0.90 ± 0.04 | ok |
| dr | ddr | site_only_head | 10 | - | - | 0.69 ± 0.09 | 0.74 ± 0.07 | - | - | - |  |
| dr | ddr | site_only_head | 50 | - | - | 0.82 ± 0.03 | 0.85 ± 0.03 | - | - | - |  |
| dr | ddr | site_only_head | 100 | - | - | 0.86 ± 0.01 | 0.89 ± 0.01 | - | - | - |  |
| dr | ddr | site_recalibration | 10 | 0.83 ± 0.11 | 0.62 ± 0.30 | 0.87 ± 0.01 | 0.90 ± 0.01 | 0.12 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | ddr | site_recalibration | 50 | 0.86 ± 0.06 | 0.65 ± 0.12 | 0.87 ± 0.01 | 0.90 ± 0.01 | 0.05 ± 0.02 | 0.93 ± 0.05 | 0.91 ± 0.05 | ok |
| dr | ddr | site_recalibration | 100 | 0.85 ± 0.05 | 0.68 ± 0.11 | 0.87 ± 0.01 | 0.90 ± 0.01 | 0.04 ± 0.02 | 0.90 ± 0.06 | 0.92 ± 0.02 | ok |
| dr | ddr | source_calibration | 0 | 0.29 ± 0.01 | 1.00 ± 0.00 | 0.87 ± 0.01 | 0.90 ± 0.01 | 0.26 ± 0.01 | 0.93 ± 0.00 | 0.44 ± 0.01 | **FAILS** pos |
| dr | idrid | site_adapt | 50 | 0.83 ± 0.06 | 0.86 ± 0.11 | 0.93 ± 0.01 | 0.97 ± 0.01 | 0.07 ± 0.02 | 0.94 ± 0.05 | 0.90 ± 0.05 | ok |
| dr | idrid | site_adapt | 100 | 0.86 ± 0.05 | 0.86 ± 0.07 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.07 ± 0.01 | 0.94 ± 0.05 | 0.92 ± 0.03 | ok |
| dr | idrid | site_only_head | 10 | - | - | 0.81 ± 0.07 | 0.88 ± 0.05 | - | - | - |  |
| dr | idrid | site_only_head | 50 | - | - | 0.93 ± 0.02 | 0.96 ± 0.01 | - | - | - |  |
| dr | idrid | site_only_head | 100 | - | - | 0.94 ± 0.02 | 0.96 ± 0.01 | - | - | - |  |
| dr | idrid | site_recalibration | 10 | 0.94 ± 0.03 | 0.55 ± 0.13 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.10 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | idrid | site_recalibration | 50 | 0.84 ± 0.06 | 0.82 ± 0.12 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.07 ± 0.01 | 0.91 ± 0.05 | 0.90 ± 0.05 | ok |
| dr | idrid | site_recalibration | 100 | 0.86 ± 0.05 | 0.80 ± 0.08 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.07 ± 0.01 | 0.92 ± 0.05 | 0.93 ± 0.03 | ok |
| dr | idrid | source_calibration | 0 | 0.92 ± 0.02 | 0.66 ± 0.03 | 0.93 ± 0.01 | 0.96 ± 0.01 | 0.10 ± 0.01 | 0.64 ± 0.03 | 0.95 ± 0.02 | **FAILS** neg |
