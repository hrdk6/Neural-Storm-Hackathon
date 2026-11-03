from clinictune.api.store import Store, _two_prop_p


def _result(decision: str, ood: bool, gate: bool = True) -> dict:
    return {
        "task": "tb",
        "image_sha256": "x" * 64,
        "decision": decision,
        "reason": "r",
        "probability": 0.5,
        "quality_gate": {"passed": gate},
        "ood": {"flagged": ood} if gate else None,
        "calibration": "site:a:v1",
        "model": {"bundle_id": "b"},
        "latency_ms": 10.0,
    }


def test_monitoring_alerts_and_daily(tmp_path):
    st = Store(tmp_path / "db.sqlite3")
    st.create_site("a", "A")
    for i in range(30):
        st.log_screening(f"r{i}", "a", _result("uncertain" if i % 2 else "refer", ood=i % 3 == 0, gate=i % 4 != 0))
    st.record_outcome("r0", 1)
    st.log_screening("missed", "a", _result("low_risk", ood=False))
    st.record_outcome("missed", 1)
    m = st.monitoring("a", "tb")
    codes = {a["code"] for a in m["alerts"]}
    assert {"unfamiliar_images", "image_quality", "many_uncertain", "missed_positive"} <= codes
    assert m["daily"] and m["daily"][-1]["n"] == 31
    assert m["missed_positives_low_risk"] == 1


def test_two_proportion_test():
    assert _two_prop_p(50, 100, 50, 100) > 0.9
    assert _two_prop_p(80, 100, 20, 100) < 1e-6
