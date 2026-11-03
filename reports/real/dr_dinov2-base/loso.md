# Leave-one-site-out results

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## DR (dinov2-base, 6278 images)

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dr | aptos2019 | site_adapt | 50 | 0.88 ± 0.04 | 0.90 ± 0.04 | 0.96 ± 0.01 | 0.94 ± 0.01 | 0.06 ± 0.03 | 0.89 ± 0.07 | 0.92 ± 0.04 | ok |
| dr | aptos2019 | site_adapt | 100 | 0.83 ± 0.06 | 0.93 ± 0.03 | 0.97 ± 0.00 | 0.95 ± 0.01 | 0.04 ± 0.01 | 0.90 ± 0.05 | 0.90 ± 0.06 | ok |
| dr | aptos2019 | site_only_head | 10 | - | - | 0.86 ± 0.10 | 0.80 ± 0.09 | - | - | - |  |
| dr | aptos2019 | site_only_head | 50 | - | - | 0.94 ± 0.01 | 0.91 ± 0.02 | - | - | - |  |
| dr | aptos2019 | site_only_head | 100 | - | - | 0.96 ± 0.01 | 0.94 ± 0.01 | - | - | - |  |
| dr | aptos2019 | site_recalibration | 10 | 0.84 ± 0.16 | 0.75 ± 0.22 | 0.94 ± 0.00 | 0.92 ± 0.00 | 0.08 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | aptos2019 | site_recalibration | 50 | 0.88 ± 0.03 | 0.84 ± 0.05 | 0.94 ± 0.00 | 0.92 ± 0.00 | 0.05 ± 0.03 | 0.89 ± 0.06 | 0.93 ± 0.04 | ok |
| dr | aptos2019 | site_recalibration | 100 | 0.84 ± 0.02 | 0.89 ± 0.02 | 0.94 ± 0.00 | 0.92 ± 0.00 | 0.03 ± 0.01 | 0.90 ± 0.04 | 0.91 ± 0.04 | ok |
| dr | aptos2019 | source_calibration | 0 | 0.98 ± 0.00 | 0.53 ± 0.01 | 0.94 ± 0.00 | 0.92 ± 0.00 | 0.24 ± 0.00 | 0.49 ± 0.01 | 0.99 ± 0.00 | **FAILS** neg |
| dr | ddr | site_adapt | 50 | 0.86 ± 0.04 | 0.69 ± 0.11 | 0.88 ± 0.01 | 0.91 ± 0.01 | 0.06 ± 0.03 | 0.93 ± 0.03 | 0.92 ± 0.03 | ok |
| dr | ddr | site_adapt | 100 | 0.84 ± 0.05 | 0.78 ± 0.09 | 0.90 ± 0.01 | 0.92 ± 0.01 | 0.05 ± 0.01 | 0.91 ± 0.04 | 0.90 ± 0.05 | ok |
| dr | ddr | site_only_head | 10 | - | - | 0.81 ± 0.07 | 0.84 ± 0.05 | - | - | - |  |
| dr | ddr | site_only_head | 50 | - | - | 0.84 ± 0.02 | 0.87 ± 0.02 | - | - | - |  |
| dr | ddr | site_only_head | 100 | - | - | 0.89 ± 0.01 | 0.91 ± 0.01 | - | - | - |  |
| dr | ddr | site_recalibration | 10 | 0.84 ± 0.10 | 0.61 ± 0.26 | 0.88 ± 0.01 | 0.91 ± 0.01 | 0.11 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | ddr | site_recalibration | 50 | 0.87 ± 0.05 | 0.61 ± 0.13 | 0.88 ± 0.01 | 0.91 ± 0.01 | 0.05 ± 0.02 | 0.94 ± 0.03 | 0.93 ± 0.04 | ok |
| dr | ddr | site_recalibration | 100 | 0.85 ± 0.04 | 0.68 ± 0.09 | 0.88 ± 0.01 | 0.91 ± 0.01 | 0.04 ± 0.01 | 0.91 ± 0.03 | 0.92 ± 0.04 | ok |
| dr | ddr | source_calibration | 0 | 0.38 ± 0.01 | 0.99 ± 0.00 | 0.88 ± 0.01 | 0.91 ± 0.01 | 0.26 ± 0.01 | 0.94 ± 0.00 | 0.42 ± 0.01 | **FAILS** pos |
| dr | idrid | site_adapt | 50 | 0.84 ± 0.07 | 0.85 ± 0.11 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.07 ± 0.02 | 0.89 ± 0.09 | 0.90 ± 0.04 | ok |
| dr | idrid | site_adapt | 100 | 0.86 ± 0.06 | 0.85 ± 0.08 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.07 ± 0.02 | 0.92 ± 0.05 | 0.92 ± 0.03 | ok |
| dr | idrid | site_only_head | 10 | - | - | 0.87 ± 0.03 | 0.92 ± 0.02 | - | - | - |  |
| dr | idrid | site_only_head | 50 | - | - | 0.93 ± 0.02 | 0.96 ± 0.01 | - | - | - |  |
| dr | idrid | site_only_head | 100 | - | - | 0.93 ± 0.01 | 0.96 ± 0.01 | - | - | - |  |
| dr | idrid | site_recalibration | 10 | 0.92 ± 0.06 | 0.65 ± 0.15 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.10 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | idrid | site_recalibration | 50 | 0.85 ± 0.07 | 0.81 ± 0.12 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.06 ± 0.01 | 0.90 ± 0.06 | 0.90 ± 0.03 | ok |
| dr | idrid | site_recalibration | 100 | 0.86 ± 0.04 | 0.83 ± 0.06 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.06 ± 0.02 | 0.88 ± 0.07 | 0.92 ± 0.03 | ok |
| dr | idrid | source_calibration | 0 | 0.93 ± 0.01 | 0.66 ± 0.03 | 0.94 ± 0.01 | 0.97 ± 0.01 | 0.08 ± 0.01 | 0.61 ± 0.02 | 0.96 ± 0.01 | **FAILS** neg |
