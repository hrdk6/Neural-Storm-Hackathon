# Leave-one-site-out results (dinov2-small)

Rows with k=0 use the training-site calibration unchanged. Every number is on the held-out half of the held-out site, which is never used for fitting. Mean ± SD over repeats.


## TB

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | tb_clinic_new | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_clinic_new | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_clinic_new | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_clinic_new | site_recalibration | 50 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.89 ± 0.04 | 0.90 ± 0.08 | ok |
| tb | tb_clinic_new | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 0.04 | **FAILS** neg |
| tb | tb_site_a | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_a | site_recalibration | 10 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_a | site_recalibration | 50 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.94 ± 0.04 | **FAILS** neg |
| tb | tb_site_a | site_recalibration | 100 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.94 ± 0.03 | **FAILS** neg |
| tb | tb_site_a | source_calibration | 0 | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.92 ± 0.02 | ok |
| tb | tb_site_b | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_b | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_b | site_recalibration | 50 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.91 ± 0.05 | 0.93 ± 0.05 | ok |
| tb | tb_site_b | site_recalibration | 100 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.71 ± 0.36 | 0.92 ± 0.07 | **FAILS** neg |
| tb | tb_site_b | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.88 ± 0.01 | 0.94 ± 0.02 | ok |
| tb | tb_site_c | site_only_head | 10 | - | - | 0.99 ± 0.02 | 0.98 ± 0.03 | - | - | - |  |
| tb | tb_site_c | site_only_head | 50 | - | - | 1.00 ± 0.01 | 0.99 ± 0.01 | - | - | - |  |
| tb | tb_site_c | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| tb | tb_site_c | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| tb | tb_site_c | site_recalibration | 50 | 1.00 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.93 ± 0.06 | 0.91 ± 0.10 | ok |
| tb | tb_site_c | site_recalibration | 100 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.93 ± 0.05 | 0.93 ± 0.10 | ok |
| tb | tb_site_c | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.81 ± 0.03 | 0.97 ± 0.01 | **FAILS** neg |


## DR

| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | cov(neg) | cov(pos) | target cov 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dr | dr_clinic_new | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_clinic_new | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_clinic_new | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_clinic_new | site_recalibration | 50 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.95 ± 0.02 | 0.96 ± 0.06 | ok |
| dr | dr_clinic_new | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.94 ± 0.02 | 0.73 ± 0.04 | **FAILS** pos |
| dr | dr_site_a | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_a | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_a | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_a | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_site_a | site_recalibration | 50 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 0.05 | 0.97 ± 0.03 | ok |
| dr | dr_site_a | site_recalibration | 100 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.93 ± 0.01 | 0.94 ± 0.03 | ok |
| dr | dr_site_a | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.97 ± 0.01 | 0.95 ± 0.01 | ok |
| dr | dr_site_b | site_only_head | 10 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_b | site_only_head | 50 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_b | site_only_head | 100 | - | - | 1.00 ± 0.00 | 1.00 ± 0.00 | - | - | - |  |
| dr | dr_site_b | site_recalibration | 10 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | ok |
| dr | dr_site_b | site_recalibration | 50 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.91 ± 0.03 | 0.88 ± 0.09 | ok |
| dr | dr_site_b | site_recalibration | 100 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.91 ± 0.02 | 0.91 ± 0.05 | ok |
| dr | dr_site_b | source_calibration | 0 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 0.00 | 1.00 ± 0.00 | ok |
