# Leave-one-site-out results

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## TB (dinov2-small, 3000 images)

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | montgomery | site_adapt | 50 | 0.86 ± 0.10 | 0.48 ± 0.17 | 0.80 ± 0.04 | 0.78 ± 0.04 | 0.13 ± 0.03 | 0.87 ± 0.12 | 0.89 ± 0.08 | ok |
| tb | montgomery | site_only_head | 10 | - | - | 0.62 ± 0.07 | 0.56 ± 0.08 | - | - | - |  |
| tb | montgomery | site_only_head | 50 | - | - | 0.81 ± 0.04 | 0.77 ± 0.06 | - | - | - |  |
| tb | montgomery | site_recalibration | 10 | 0.85 ± 0.13 | 0.23 ± 0.15 | 0.59 ± 0.04 | 0.51 ± 0.04 | 0.13 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | montgomery | site_recalibration | 50 | 0.85 ± 0.12 | 0.22 ± 0.11 | 0.59 ± 0.04 | 0.51 ± 0.04 | 0.07 ± 0.04 | 0.91 ± 0.08 | 0.90 ± 0.08 | ok |
| tb | montgomery | source_calibration | 0 | 0.77 ± 0.04 | 0.39 ± 0.05 | 0.59 ± 0.04 | 0.51 ± 0.04 | 0.41 ± 0.03 | 0.29 ± 0.04 | 0.77 ± 0.04 | **FAILS** neg,pos |
| tb | shenzhen | site_adapt | 50 | 0.91 ± 0.04 | 0.47 ± 0.13 | 0.84 ± 0.02 | 0.85 ± 0.02 | 0.08 ± 0.02 | 0.92 ± 0.06 | 0.93 ± 0.04 | ok |
| tb | shenzhen | site_adapt | 100 | 0.91 ± 0.07 | 0.49 ± 0.20 | 0.86 ± 0.02 | 0.87 ± 0.02 | 0.08 ± 0.03 | 0.92 ± 0.05 | 0.92 ± 0.04 | ok |
| tb | shenzhen | site_only_head | 10 | - | - | 0.83 ± 0.02 | 0.84 ± 0.03 | - | - | - |  |
| tb | shenzhen | site_only_head | 50 | - | - | 0.85 ± 0.02 | 0.86 ± 0.02 | - | - | - |  |
| tb | shenzhen | site_only_head | 100 | - | - | 0.86 ± 0.01 | 0.88 ± 0.01 | - | - | - |  |
| tb | shenzhen | site_recalibration | 10 | 0.86 ± 0.14 | 0.23 ± 0.17 | 0.58 ± 0.02 | 0.58 ± 0.02 | 0.13 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | shenzhen | site_recalibration | 50 | 0.91 ± 0.04 | 0.21 ± 0.07 | 0.58 ± 0.02 | 0.58 ± 0.02 | 0.08 ± 0.03 | 0.90 ± 0.06 | 0.94 ± 0.03 | ok |
| tb | shenzhen | site_recalibration | 100 | 0.90 ± 0.02 | 0.23 ± 0.05 | 0.58 ± 0.02 | 0.58 ± 0.02 | 0.06 ± 0.03 | 0.91 ± 0.03 | 0.91 ± 0.03 | ok |
| tb | shenzhen | source_calibration | 0 | 0.62 ± 0.02 | 0.49 ± 0.03 | 0.58 ± 0.02 | 0.58 ± 0.02 | 0.26 ± 0.01 | 0.42 ± 0.03 | 0.62 ± 0.02 | **FAILS** neg,pos |
| tb | tbx11k | site_adapt | 50 | 0.85 ± 0.10 | 0.77 ± 0.12 | 0.91 ± 0.02 | 0.86 ± 0.03 | 0.06 ± 0.02 | 0.92 ± 0.04 | 0.91 ± 0.05 | ok |
| tb | tbx11k | site_adapt | 100 | 0.94 ± 0.03 | 0.72 ± 0.08 | 0.94 ± 0.01 | 0.90 ± 0.01 | 0.04 ± 0.01 | 0.92 ± 0.03 | 0.95 ± 0.03 | ok |
| tb | tbx11k | site_only_head | 10 | - | - | 0.77 ± 0.09 | 0.65 ± 0.14 | - | - | - |  |
| tb | tbx11k | site_only_head | 50 | - | - | 0.91 ± 0.02 | 0.86 ± 0.03 | - | - | - |  |
| tb | tbx11k | site_only_head | 100 | - | - | 0.94 ± 0.01 | 0.90 ± 0.01 | - | - | - |  |
| tb | tbx11k | site_recalibration | 10 | 0.83 ± 0.11 | 0.27 ± 0.17 | 0.61 ± 0.01 | 0.45 ± 0.01 | 0.14 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tbx11k | site_recalibration | 50 | 0.93 ± 0.05 | 0.13 ± 0.08 | 0.61 ± 0.01 | 0.45 ± 0.01 | 0.03 ± 0.02 | 0.93 ± 0.04 | 0.97 ± 0.03 | ok |
| tb | tbx11k | site_recalibration | 100 | 0.91 ± 0.05 | 0.17 ± 0.07 | 0.61 ± 0.01 | 0.45 ± 0.01 | 0.03 ± 0.01 | 0.93 ± 0.03 | 0.92 ± 0.04 | ok |
| tb | tbx11k | source_calibration | 0 | 0.97 ± 0.00 | 0.05 ± 0.00 | 0.61 ± 0.01 | 0.45 ± 0.01 | 0.36 ± 0.01 | 0.25 ± 0.01 | 0.97 ± 0.00 | **FAILS** neg |
