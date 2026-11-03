# Leave-one-site-out results

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## TB (dinov2-base, 3000 images)

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | montgomery | site_adapt | 50 | 0.83 ± 0.11 | 0.44 ± 0.17 | 0.76 ± 0.05 | 0.73 ± 0.07 | 0.13 ± 0.04 | 0.92 ± 0.09 | 0.90 ± 0.10 | ok |
| tb | montgomery | site_only_head | 10 | - | - | 0.65 ± 0.06 | 0.58 ± 0.04 | - | - | - |  |
| tb | montgomery | site_only_head | 50 | - | - | 0.79 ± 0.03 | 0.77 ± 0.05 | - | - | - |  |
| tb | montgomery | site_recalibration | 10 | 0.79 ± 0.13 | 0.36 ± 0.23 | 0.70 ± 0.03 | 0.63 ± 0.06 | 0.16 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | montgomery | site_recalibration | 50 | 0.86 ± 0.10 | 0.25 ± 0.11 | 0.70 ± 0.03 | 0.63 ± 0.06 | 0.14 ± 0.03 | 0.94 ± 0.05 | 0.90 ± 0.08 | ok |
| tb | montgomery | source_calibration | 0 | 0.65 ± 0.05 | 0.71 ± 0.04 | 0.70 ± 0.03 | 0.63 ± 0.06 | 0.22 ± 0.03 | 0.63 ± 0.05 | 0.67 ± 0.05 | **FAILS** neg,pos |
| tb | shenzhen | site_adapt | 50 | 0.90 ± 0.03 | 0.55 ± 0.09 | 0.85 ± 0.02 | 0.86 ± 0.03 | 0.08 ± 0.02 | 0.92 ± 0.09 | 0.92 ± 0.02 | ok |
| tb | shenzhen | site_adapt | 100 | 0.92 ± 0.04 | 0.55 ± 0.19 | 0.87 ± 0.02 | 0.88 ± 0.02 | 0.07 ± 0.02 | 0.91 ± 0.05 | 0.94 ± 0.04 | ok |
| tb | shenzhen | site_only_head | 10 | - | - | 0.83 ± 0.03 | 0.83 ± 0.06 | - | - | - |  |
| tb | shenzhen | site_only_head | 50 | - | - | 0.86 ± 0.02 | 0.87 ± 0.02 | - | - | - |  |
| tb | shenzhen | site_only_head | 100 | - | - | 0.87 ± 0.02 | 0.88 ± 0.02 | - | - | - |  |
| tb | shenzhen | site_recalibration | 10 | 0.93 ± 0.05 | 0.24 ± 0.11 | 0.67 ± 0.02 | 0.64 ± 0.02 | 0.10 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | shenzhen | site_recalibration | 50 | 0.89 ± 0.06 | 0.33 ± 0.09 | 0.67 ± 0.02 | 0.64 ± 0.02 | 0.07 ± 0.02 | 0.94 ± 0.05 | 0.93 ± 0.05 | ok |
| tb | shenzhen | site_recalibration | 100 | 0.89 ± 0.04 | 0.32 ± 0.04 | 0.67 ± 0.02 | 0.64 ± 0.02 | 0.06 ± 0.03 | 0.90 ± 0.04 | 0.90 ± 0.04 | ok |
| tb | shenzhen | source_calibration | 0 | 0.97 ± 0.01 | 0.18 ± 0.03 | 0.67 ± 0.02 | 0.64 ± 0.02 | 0.26 ± 0.01 | 0.08 ± 0.01 | 0.97 ± 0.01 | **FAILS** neg |
| tb | tbx11k | site_adapt | 50 | 0.85 ± 0.07 | 0.75 ± 0.10 | 0.89 ± 0.02 | 0.85 ± 0.02 | 0.05 ± 0.02 | 0.92 ± 0.03 | 0.92 ± 0.06 | ok |
| tb | tbx11k | site_adapt | 100 | 0.92 ± 0.04 | 0.75 ± 0.10 | 0.94 ± 0.01 | 0.91 ± 0.02 | 0.05 ± 0.02 | 0.93 ± 0.04 | 0.94 ± 0.04 | ok |
| tb | tbx11k | site_only_head | 10 | - | - | 0.78 ± 0.07 | 0.65 ± 0.11 | - | - | - |  |
| tb | tbx11k | site_only_head | 50 | - | - | 0.89 ± 0.03 | 0.84 ± 0.03 | - | - | - |  |
| tb | tbx11k | site_only_head | 100 | - | - | 0.94 ± 0.01 | 0.90 ± 0.02 | - | - | - |  |
| tb | tbx11k | site_recalibration | 10 | 0.83 ± 0.11 | 0.35 ± 0.14 | 0.67 ± 0.01 | 0.54 ± 0.01 | 0.12 ± 0.07 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tbx11k | site_recalibration | 50 | 0.90 ± 0.04 | 0.26 ± 0.07 | 0.67 ± 0.01 | 0.54 ± 0.01 | 0.07 ± 0.05 | 0.89 ± 0.07 | 0.95 ± 0.04 | ok |
| tb | tbx11k | site_recalibration | 100 | 0.94 ± 0.04 | 0.18 ± 0.07 | 0.67 ± 0.01 | 0.54 ± 0.01 | 0.03 ± 0.01 | 0.93 ± 0.03 | 0.95 ± 0.04 | ok |
| tb | tbx11k | source_calibration | 0 | 0.98 ± 0.01 | 0.06 ± 0.01 | 0.67 ± 0.01 | 0.54 ± 0.01 | 0.26 ± 0.01 | 0.41 ± 0.01 | 0.98 ± 0.01 | **FAILS** neg |
