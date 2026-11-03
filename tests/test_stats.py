import numpy as np
import pytest

from clinictune.data.splits import (
    LeakageError,
    assert_no_patient_leakage,
    patient_split,
    stratified_label_sample,
)
from clinictune.model.calibration import Platt, clopper_pearson, fit_platt, threshold_for_sensitivity
from clinictune.model.conformal import ClassConditionalConformal, min_per_class
from clinictune.model.heads import fit_head
from clinictune.model.metrics import binary_report
from clinictune.triage import Decision, decide


def _scores(n, rng, prevalence=0.3):
    y = (rng.random(n) < prevalence).astype(int)
    z = rng.normal(np.where(y == 1, 1.2, -1.2), 1.0)
    return 1 / (1 + np.exp(-z)), y


def test_class_conditional_coverage_holds_per_class():
    rng = np.random.default_rng(0)
    alpha = 0.1
    cov = {0: [], 1: []}
    for _ in range(200):
        p_cal, y_cal = _scores(300, rng)
        p_te, y_te = _scores(300, rng)
        c = ClassConditionalConformal.fit(p_cal, y_cal, alpha)
        sets = c.predict_sets(p_te)
        for k in (0, 1):
            cov[k].append(sets[y_te == k, k].mean())
    # The guarantee is per class, in expectation over calibration draws.
    assert np.mean(cov[0]) >= 1 - alpha - 0.01
    assert np.mean(cov[1]) >= 1 - alpha - 0.01


def test_conformal_too_few_labels_never_excludes_class():
    alpha = 0.1
    n_pos = min_per_class(alpha) - 1
    p = np.r_[np.full(50, 0.1), np.full(n_pos, 0.9)]
    y = np.r_[np.zeros(50), np.ones(n_pos)].astype(int)
    c = ClassConditionalConformal.fit(p, y, alpha)
    assert c.qhat[1] is None and not c.is_informative
    assert c.predict_sets(np.array([0.0, 0.5, 1.0]))[:, 1].all()
    assert ClassConditionalConformal.from_dict(c.to_dict()).qhat == c.qhat


def test_platt_recovers_and_prior_shrinks():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(0.5 * z - 1.0)))).astype(int)
    fit = fit_platt(z, y)
    assert abs(fit.a - 0.5) < 0.1 and abs(fit.b + 1.0) < 0.15
    prior = Platt(1.0, 0.0)
    small = fit_platt(z[:10], y[:10], prior=prior, prior_strength=5.0)
    free = fit_platt(z[:10], y[:10])
    d = lambda p: abs(p.a - prior.a) + abs(p.b - prior.b)  # noqa: E731
    assert d(small) <= d(free)


def test_threshold_meets_sensitivity():
    rng = np.random.default_rng(2)
    p, y = _scores(500, rng)
    t = threshold_for_sensitivity(p, y, 0.9)
    assert (p[y == 1] >= t).mean() >= 0.9


def test_clopper_pearson_bounds():
    lo, hi = clopper_pearson(9, 10)
    assert 0.55 < lo < 0.6 and 0.99 < hi <= 1.0
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_patient_split_no_leakage_and_detection():
    rng = np.random.default_rng(3)
    patients = np.repeat(np.arange(200), 2).astype(str)
    y = np.repeat((rng.random(200) < 0.3).astype(int), 2)
    parts = patient_split(y, patients, [0.6, 0.2, 0.2], seed=0)
    assert sum(len(p) for p in parts) == len(y)
    assert all(len(p) > 0 for p in parts)
    with pytest.raises(LeakageError):
        assert_no_patient_leakage(["a", "b"], ["b", "c"])
    idx = stratified_label_sample(y, patients, 20, rng)
    assert set(y[idx]) == {0, 1}


def test_head_token_decomposition_is_exact():
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(60, 49, 16)).astype(np.float32)
    y = (tokens.mean(1)[:, 0] > 0).astype(int)
    head, _ = fit_head(tokens.mean(1), y, np.arange(60).astype(str))
    contrib = head.token_contributions(tokens[0])
    assert np.isclose(contrib.mean() + head.intercept, head.logit(tokens[0].mean(0)), atol=1e-4)


def test_binary_report_fields():
    rng = np.random.default_rng(5)
    p, y = _scores(200, rng)
    r = binary_report(y, p, 0.5, n_boot=50)
    for k in ("sensitivity", "specificity", "roc_auc", "pr_auc", "ece", "roc_auc_ci", "sensitivity_ci"):
        assert r[k] is not None


@pytest.mark.parametrize(
    "p,in_set,expected",
    [
        (0.9, [False, True], Decision.REFER),
        (0.6, [True, True], Decision.REFER),
        (0.2, [True, True], Decision.UNCERTAIN),
        (0.1, [True, False], Decision.LOW_RISK),
        (0.6, [True, False], Decision.UNCERTAIN),
        (0.5, [False, False], Decision.UNCERTAIN),
    ],
)
def test_decision_rule(p, in_set, expected):
    assert decide(p, np.array(in_set), threshold=0.5)[0] == expected
