# Leave-one-site-out results (demo-randproj)

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## TB

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | tb_clinic_new | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_clinic_new | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_clinic_new | site_recalibration | 10 | 0.84 ± 0.21 | 0.34 ± 0.29 | 0.79 ± 0.04 | 0.66 ± 0.06 | 0.10 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_clinic_new | site_recalibration | 50 | 0.97 ± 0.02 | 0.14 ± 0.09 | 0.79 ± 0.04 | 0.66 ± 0.06 | 0.04 ± 0.01 | 0.95 ± 0.03 | 0.97 ± 0.02 | ok |
| tb | tb_clinic_new | source_calibration | 0 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.79 ± 0.04 | 0.66 ± 0.06 | 0.26 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | **FAILS** pos |
| tb | tb_site_a | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_recalibration | 10 | 0.79 ± 0.12 | 0.98 ± 0.04 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.23 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_a | site_recalibration | 50 | 0.86 ± 0.09 | 0.93 ± 0.07 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.09 ± 0.02 | 0.91 ± 0.05 | 0.93 ± 0.04 | ok |
| tb | tb_site_a | site_recalibration | 100 | 0.88 ± 0.06 | 0.95 ± 0.03 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.06 ± 0.01 | 0.91 ± 0.04 | 0.89 ± 0.06 | ok |
| tb | tb_site_a | source_calibration | 0 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.98 ± 0.00 | 0.97 ± 0.00 | 0.64 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | **FAILS** neg |
| tb | tb_site_b | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_recalibration | 10 | 0.88 ± 0.04 | 0.86 ± 0.15 | 0.95 ± 0.01 | 0.93 ± 0.01 | 0.31 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_b | site_recalibration | 50 | 0.93 ± 0.07 | 0.71 ± 0.25 | 0.95 ± 0.01 | 0.93 ± 0.01 | 0.20 ± 0.01 | 0.91 ± 0.05 | 0.95 ± 0.05 | ok |
| tb | tb_site_b | site_recalibration | 100 | 0.91 ± 0.04 | 0.82 ± 0.10 | 0.95 ± 0.01 | 0.93 ± 0.01 | 0.10 ± 0.01 | 0.88 ± 0.07 | 0.92 ± 0.04 | ok |
| tb | tb_site_b | source_calibration | 0 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.95 ± 0.01 | 0.93 ± 0.01 | 0.73 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 | **FAILS** neg |
| tb | tb_site_c | site_only_head | 10 | - | - | 0.97 ± 0.01 | 0.96 ± 0.01 | - | - | - |  |
| tb | tb_site_c | site_only_head | 50 | - | - | 0.98 ± 0.01 | 0.97 ± 0.02 | - | - | - |  |
| tb | tb_site_c | site_only_head | 100 | - | - | 0.99 ± 0.01 | 0.99 ± 0.01 | - | - | - |  |
| tb | tb_site_c | site_recalibration | 10 | 0.92 ± 0.07 | 0.83 ± 0.01 | 0.96 ± 0.00 | 0.95 ± 0.01 | 0.23 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_c | site_recalibration | 50 | 0.89 ± 0.09 | 0.85 ± 0.04 | 0.96 ± 0.00 | 0.95 ± 0.01 | 0.17 ± 0.03 | 0.88 ± 0.07 | 0.87 ± 0.09 | ok |
| tb | tb_site_c | site_recalibration | 100 | 0.92 ± 0.05 | 0.84 ± 0.02 | 0.96 ± 0.00 | 0.95 ± 0.01 | 0.15 ± 0.01 | 0.93 ± 0.02 | 0.90 ± 0.06 | ok |
| tb | tb_site_c | source_calibration | 0 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.96 ± 0.00 | 0.95 ± 0.01 | 0.39 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | **FAILS** pos |


## DR

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dr | dr_clinic_new | site_only_head | 10 | - | - | 0.90 ± 0.02 | 0.85 ± 0.03 | - | - | - |  |
| dr | dr_clinic_new | site_only_head | 50 | - | - | 0.95 ± 0.01 | 0.92 ± 0.02 | - | - | - |  |
| dr | dr_clinic_new | site_recalibration | 10 | 0.61 ± 0.15 | 0.90 ± 0.12 | 0.88 ± 0.03 | 0.84 ± 0.03 | 0.11 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_clinic_new | site_recalibration | 50 | 0.79 ± 0.14 | 0.75 ± 0.18 | 0.88 ± 0.03 | 0.84 ± 0.03 | 0.11 ± 0.05 | 0.90 ± 0.05 | 0.89 ± 0.10 | ok |
| dr | dr_clinic_new | source_calibration | 0 | 0.99 ± 0.02 | 0.12 ± 0.04 | 0.88 ± 0.03 | 0.84 ± 0.03 | 0.30 ± 0.03 | 0.30 ± 0.05 | 1.00 ± 0.00 | **FAILS** neg |
| dr | dr_site_a | site_only_head | 10 | - | - | 0.96 ± 0.02 | 0.94 ± 0.02 | - | - | - |  |
| dr | dr_site_a | site_only_head | 50 | - | - | 0.99 ± 0.00 | 0.98 ± 0.01 | - | - | - |  |
| dr | dr_site_a | site_only_head | 100 | - | - | 0.99 ± 0.00 | 0.99 ± 0.00 | - | - | - |  |
| dr | dr_site_a | site_recalibration | 10 | 0.82 ± 0.24 | 0.67 ± 0.24 | 0.93 ± 0.01 | 0.90 ± 0.01 | 0.10 ± 0.05 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_site_a | site_recalibration | 50 | 0.84 ± 0.02 | 0.89 ± 0.01 | 0.93 ± 0.01 | 0.90 ± 0.01 | 0.07 ± 0.01 | 0.95 ± 0.02 | 0.94 ± 0.04 | ok |
| dr | dr_site_a | site_recalibration | 100 | 0.84 ± 0.04 | 0.88 ± 0.03 | 0.93 ± 0.01 | 0.90 ± 0.01 | 0.07 ± 0.02 | 0.93 ± 0.04 | 0.89 ± 0.05 | ok |
| dr | dr_site_a | source_calibration | 0 | 0.49 ± 0.06 | 0.99 ± 0.01 | 0.93 ± 0.01 | 0.90 ± 0.01 | 0.18 ± 0.02 | 1.00 ± 0.00 | 0.88 ± 0.03 | ok |
| dr | dr_site_b | site_only_head | 10 | - | - | 0.94 ± 0.02 | 0.87 ± 0.04 | - | - | - |  |
| dr | dr_site_b | site_only_head | 50 | - | - | 0.98 ± 0.01 | 0.94 ± 0.02 | - | - | - |  |
| dr | dr_site_b | site_only_head | 100 | - | - | 0.98 ± 0.01 | 0.95 ± 0.02 | - | - | - |  |
| dr | dr_site_b | site_recalibration | 10 | 0.88 ± 0.07 | 0.78 ± 0.16 | 0.93 ± 0.02 | 0.84 ± 0.03 | 0.12 ± 0.06 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_site_b | site_recalibration | 50 | 0.74 ± 0.15 | 0.87 ± 0.07 | 0.93 ± 0.02 | 0.84 ± 0.03 | 0.06 ± 0.01 | 0.90 ± 0.05 | 0.85 ± 0.14 | ok |
| dr | dr_site_b | site_recalibration | 100 | 0.83 ± 0.07 | 0.83 ± 0.03 | 0.93 ± 0.02 | 0.84 ± 0.03 | 0.06 ± 0.02 | 0.89 ± 0.07 | 0.93 ± 0.02 | ok |
| dr | dr_site_b | source_calibration | 0 | 0.99 ± 0.01 | 0.47 ± 0.04 | 0.93 ± 0.02 | 0.84 ± 0.03 | 0.28 ± 0.02 | 0.58 ± 0.04 | 1.00 ± 0.00 | **FAILS** neg |
