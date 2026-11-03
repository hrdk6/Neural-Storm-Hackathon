# Leave-one-site-out results

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## TB (rad-dino@392, 3000 images)

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | montgomery | site_adapt | 50 | 0.88 ± 0.10 | 0.59 ± 0.28 | 0.89 ± 0.04 | 0.89 ± 0.04 | 0.14 ± 0.04 | 0.92 ± 0.10 | 0.91 ± 0.11 | ok |
| tb | montgomery | site_only_head | 10 | - | - | 0.71 ± 0.06 | 0.65 ± 0.08 | - | - | - |  |
| tb | montgomery | site_only_head | 50 | - | - | 0.89 ± 0.03 | 0.88 ± 0.03 | - | - | - |  |
| tb | montgomery | site_recalibration | 10 | 0.76 ± 0.18 | 0.51 ± 0.22 | 0.73 ± 0.03 | 0.65 ± 0.05 | 0.12 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | montgomery | site_recalibration | 50 | 0.89 ± 0.10 | 0.34 ± 0.17 | 0.73 ± 0.03 | 0.65 ± 0.05 | 0.11 ± 0.02 | 0.93 ± 0.09 | 0.91 ± 0.09 | ok |
| tb | montgomery | source_calibration | 0 | 0.97 ± 0.01 | 0.14 ± 0.04 | 0.73 ± 0.03 | 0.65 ± 0.05 | 0.54 ± 0.01 | 0.04 ± 0.02 | 0.97 ± 0.01 | **FAILS** neg |
| tb | shenzhen | site_adapt | 50 | 0.92 ± 0.03 | 0.74 ± 0.12 | 0.93 ± 0.01 | 0.94 ± 0.02 | 0.08 ± 0.03 | 0.89 ± 0.10 | 0.94 ± 0.04 | ok |
| tb | shenzhen | site_adapt | 100 | 0.90 ± 0.04 | 0.81 ± 0.09 | 0.94 ± 0.01 | 0.95 ± 0.01 | 0.06 ± 0.02 | 0.92 ± 0.05 | 0.90 ± 0.04 | ok |
| tb | shenzhen | site_only_head | 10 | - | - | 0.85 ± 0.02 | 0.85 ± 0.04 | - | - | - |  |
| tb | shenzhen | site_only_head | 50 | - | - | 0.91 ± 0.02 | 0.91 ± 0.02 | - | - | - |  |
| tb | shenzhen | site_only_head | 100 | - | - | 0.94 ± 0.01 | 0.95 ± 0.01 | - | - | - |  |
| tb | shenzhen | site_recalibration | 10 | 0.89 ± 0.09 | 0.45 ± 0.20 | 0.82 ± 0.01 | 0.84 ± 0.01 | 0.09 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | shenzhen | site_recalibration | 50 | 0.91 ± 0.05 | 0.39 ± 0.14 | 0.82 ± 0.01 | 0.84 ± 0.01 | 0.07 ± 0.02 | 0.93 ± 0.04 | 0.93 ± 0.04 | ok |
| tb | shenzhen | site_recalibration | 100 | 0.89 ± 0.03 | 0.47 ± 0.10 | 0.82 ± 0.01 | 0.84 ± 0.01 | 0.06 ± 0.03 | 0.88 ± 0.07 | 0.90 ± 0.03 | ok |
| tb | shenzhen | source_calibration | 0 | 0.72 ± 0.02 | 0.77 ± 0.04 | 0.82 ± 0.01 | 0.84 ± 0.01 | 0.15 ± 0.02 | 0.31 ± 0.04 | 0.72 ± 0.02 | **FAILS** neg,pos |
| tb | tbx11k | site_adapt | 50 | 0.89 ± 0.06 | 0.82 ± 0.12 | 0.95 ± 0.02 | 0.92 ± 0.02 | 0.04 ± 0.02 | 0.93 ± 0.04 | 0.95 ± 0.06 | ok |
| tb | tbx11k | site_adapt | 100 | 0.95 ± 0.02 | 0.82 ± 0.09 | 0.97 ± 0.01 | 0.96 ± 0.01 | 0.05 ± 0.02 | 0.94 ± 0.04 | 0.96 ± 0.03 | ok |
| tb | tbx11k | site_only_head | 10 | - | - | 0.81 ± 0.06 | 0.71 ± 0.12 | - | - | - |  |
| tb | tbx11k | site_only_head | 50 | - | - | 0.93 ± 0.03 | 0.91 ± 0.04 | - | - | - |  |
| tb | tbx11k | site_only_head | 100 | - | - | 0.97 ± 0.01 | 0.96 ± 0.01 | - | - | - |  |
| tb | tbx11k | site_recalibration | 10 | 0.80 ± 0.05 | 0.66 ± 0.09 | 0.82 ± 0.00 | 0.73 ± 0.01 | 0.12 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tbx11k | site_recalibration | 50 | 0.87 ± 0.07 | 0.56 ± 0.11 | 0.82 ± 0.00 | 0.73 ± 0.01 | 0.05 ± 0.02 | 0.91 ± 0.04 | 0.93 ± 0.06 | ok |
| tb | tbx11k | site_recalibration | 100 | 0.91 ± 0.04 | 0.50 ± 0.08 | 0.82 ± 0.00 | 0.73 ± 0.01 | 0.05 ± 0.01 | 0.93 ± 0.03 | 0.94 ± 0.04 | ok |
| tb | tbx11k | source_calibration | 0 | 0.87 ± 0.01 | 0.59 ± 0.01 | 0.82 ± 0.00 | 0.73 ± 0.01 | 0.19 ± 0.00 | 0.48 ± 0.01 | 0.83 ± 0.01 | **FAILS** neg,pos |
