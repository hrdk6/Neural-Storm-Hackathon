"""SQLite persistence: sites, versioned site calibrations + report cards, onboarding jobs, audit log.

No images and no patient identifiers are stored - only the image sha256 and the screening outcome.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from clinictune.site.onboarding import SiteCalibration

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sites (
    site_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibrations (
    site_id TEXT NOT NULL REFERENCES sites(site_id),
    task TEXT NOT NULL,
    version INTEGER NOT NULL,
    bundle_id TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    calibration TEXT NOT NULL,
    report TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (site_id, task, version)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screenings (
    request_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    site_id TEXT,
    task TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    probability REAL,
    gate_passed INTEGER NOT NULL,
    ood_flagged INTEGER,
    calibration TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    latency_ms REAL,
    outcome INTEGER
);
CREATE INDEX IF NOT EXISTS screenings_site_ts ON screenings(site_id, task, ts);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def jsonable(obj: Any) -> Any:
    """Recursively replace NaN/inf (invalid JSON) with None and tuples with lists."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return jsonable(obj.item())
    return obj


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- sites
    def create_site(self, site_id: str, name: str) -> dict[str, Any]:
        with self._conn() as c:
            c.execute("INSERT INTO sites VALUES (?, ?, ?)", (site_id, name, now()))
        return self.get_site(site_id)  # type: ignore[return-value]

    def get_site(self, site_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,)).fetchone()
            if row is None:
                return None
            cals = c.execute(
                "SELECT task, version, bundle_id, active, created_at, json_extract(report, '$.verdict') AS verdict "
                "FROM calibrations WHERE site_id = ? ORDER BY task, version",
                (site_id,),
            ).fetchall()
        return {**dict(row), "calibrations": [dict(r) for r in cals]}

    def list_sites(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT site_id FROM sites ORDER BY created_at")]
        return [s for s in (self.get_site(i) for i in ids) if s]

    # ---- calibrations
    def add_calibration(self, calib: SiteCalibration, report: dict[str, Any], activate: bool = True) -> int:
        with self._lock, self._conn() as c:
            (v,) = c.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM calibrations WHERE site_id = ? AND task = ?",
                (calib.site_id, calib.task),
            ).fetchone()
            calib.version = v
            report = {**report, "version": v}
            if activate:
                c.execute(
                    "UPDATE calibrations SET active = 0 WHERE site_id = ? AND task = ?", (calib.site_id, calib.task)
                )
            c.execute(
                "INSERT INTO calibrations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    calib.site_id,
                    calib.task,
                    v,
                    calib.bundle_id,
                    int(activate),
                    json.dumps(jsonable(calib.to_dict())),
                    json.dumps(jsonable(report)),
                    now(),
                ),
            )
        return int(v)

    def activate(self, site_id: str, task: str, version: int) -> bool:
        with self._lock, self._conn() as c:
            exists = c.execute(
                "SELECT 1 FROM calibrations WHERE site_id=? AND task=? AND version=?", (site_id, task, version)
            ).fetchone()
            if not exists:
                return False
            c.execute(
                "UPDATE calibrations SET active = (version = ?) WHERE site_id = ? AND task = ?",
                (version, site_id, task),
            )
        return True

    def active_calibration(self, site_id: str, task: str) -> SiteCalibration | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT calibration FROM calibrations WHERE site_id=? AND task=? AND active=1", (site_id, task)
            ).fetchone()
        return SiteCalibration.from_dict(json.loads(row[0])) if row else None

    def report(self, site_id: str, task: str, version: int | None = None) -> dict[str, Any] | None:
        q = "SELECT report FROM calibrations WHERE site_id=? AND task=? AND "
        q += "version=?" if version is not None else "active=1"
        args: tuple = (site_id, task, version) if version is not None else (site_id, task)
        with self._conn() as c:
            row = c.execute(q, args).fetchone()
        return json.loads(row[0]) if row else None

    # ---- jobs
    def create_job(self, site_id: str, task: str) -> str:
        job_id = uuid.uuid4().hex
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, 'queued', 0, NULL, NULL, ?, ?)",
                (job_id, site_id, task, now(), now()),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict | None = None,
    ) -> None:
        sets, args = ["updated_at = ?"], [now()]
        for col, val in (("status", status), ("progress", progress), ("message", message)):
            if val is not None:
                sets.append(f"{col} = ?")
                args.append(val)
        if result is not None:
            sets.append("result = ?")
            args.append(json.dumps(jsonable(result)))
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", (*args, job_id))  # noqa: S608

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def fail_stale_jobs(self) -> None:
        """Jobs left running by a crashed/restarted process can never finish."""
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='failed', message='server restarted during job' "
                "WHERE status IN ('queued', 'running')"
            )

    # ---- audit log & monitoring
    def log_screening(self, request_id: str, site_id: str | None, r: dict[str, Any]) -> None:
        ood = r.get("ood") or {}
        with self._conn() as c:
            c.execute(
                "INSERT INTO screenings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    request_id,
                    now(),
                    site_id,
                    r["task"],
                    r["image_sha256"],
                    str(r["decision"]),
                    r.get("reason"),
                    r.get("probability"),
                    int(r["quality_gate"]["passed"]),
                    None if not ood else int(ood["flagged"]),
                    r["calibration"],
                    r["model"]["bundle_id"],
                    r.get("latency_ms"),
                ),
            )

    def record_outcome(self, request_id: str, outcome: int) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE screenings SET outcome = ? WHERE request_id = ?", (outcome, request_id))
        return cur.rowcount == 1

    def monitoring(self, site_id: str, task: str, days: int = 30) -> dict[str, Any]:
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            rows = [
                dict(r)
                for r in c.execute(
                    "SELECT ts, decision, gate_passed, ood_flagged, outcome, probability, latency_ms, calibration "
                    "FROM screenings WHERE site_id = ? AND task = ? AND ts >= ? ORDER BY ts",
                    (site_id, task, since),
                )
            ]
        n = len(rows)
        out: dict[str, Any] = {"site_id": site_id, "task": task, "days": days, "n": n, "alerts": [], "daily": []}
        if not n:
            return out
        dec = [r["decision"] for r in rows]
        out["decision_rates"] = {d: dec.count(d) / n for d in ("refer", "low_risk", "uncertain")}
        out["gate_fail_rate"] = sum(1 - r["gate_passed"] for r in rows) / n
        flagged = [r["ood_flagged"] for r in rows if r["ood_flagged"] is not None]
        out["ood_rate"] = sum(flagged) / len(flagged) if flagged else None
        lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
        out["latency_ms_p50"] = lat[len(lat) // 2] if lat else None
        confirmed = [r for r in rows if r["outcome"] is not None]
        out["confirmed_outcomes"] = len(confirmed)
        pos = [r for r in confirmed if r["outcome"] == 1]
        neg = [r for r in confirmed if r["outcome"] == 0]
        if pos:
            out["observed_sensitivity"] = sum(r["decision"] == "refer" for r in pos) / len(pos)
            out["missed_positives_low_risk"] = sum(r["decision"] == "low_risk" for r in pos)
        if neg:
            out["observed_specificity"] = sum(r["decision"] == "low_risk" for r in neg) / len(neg)
        out["daily"] = _daily(rows)
        out["alerts"] = _alerts(rows, out)
        out["note"] = (
            "Outcomes are only recorded for patients whose confirmatory test result was entered; "
            "observed rates are biased if confirmation is not done for every patient."
        )
        return out


def _daily(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days: dict[str, dict[str, int]] = {}
    for r in rows:
        d = days.setdefault(r["ts"][:10], {"n": 0, "refer": 0, "low_risk": 0, "uncertain": 0, "ood": 0, "gate_fail": 0})
        d["n"] += 1
        d[r["decision"]] = d.get(r["decision"], 0) + 1
        d["ood"] += int(r["ood_flagged"] or 0)
        d["gate_fail"] += 1 - r["gate_passed"]
    return [{"date": k, **v} for k, v in sorted(days.items())]


def _two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value of a two-proportion z-test."""
    from math import erf, sqrt

    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(k1 / n1 - k2 / n2) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def _alerts(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, str]]:
    """Plain rules a clinic lead can act on. Thresholds are deliberately simple and documented."""
    alerts = []
    n = len(rows)
    if n >= 20 and (summary.get("ood_rate") or 0) > 0.10:
        alerts.append(
            {
                "level": "warning",
                "code": "unfamiliar_images",
                "message": f"{summary['ood_rate']:.0%} of images look unlike the data this site was "
                "calibrated on. Has the X-ray machine, camera or protocol changed? Re-onboard.",
            }
        )
    if n >= 20 and summary["gate_fail_rate"] > 0.20:
        alerts.append(
            {
                "level": "warning",
                "code": "image_quality",
                "message": f"{summary['gate_fail_rate']:.0%} of images fail the quality gate. Check focus, "
                "exposure and positioning, or camera hardware.",
            }
        )
    if n >= 20 and summary["decision_rates"]["uncertain"] > 0.40:
        alerts.append(
            {
                "level": "info",
                "code": "many_uncertain",
                "message": f"{summary['decision_rates']['uncertain']:.0%} of results are Uncertain. More local "
                "labels at the next onboarding will make the model more decisive.",
            }
        )
    recent = [r for r in rows if r["ts"] >= (datetime.now(UTC) - timedelta(days=7)).isoformat(timespec="seconds")]
    base = [r for r in rows if r not in recent]
    if len(recent) >= 30 and len(base) >= 30:
        k1, k2 = sum(r["decision"] == "refer" for r in recent), sum(r["decision"] == "refer" for r in base)
        p = _two_prop_p(k1, len(recent), k2, len(base))
        if p < 0.01:
            alerts.append(
                {
                    "level": "warning",
                    "code": "referral_rate_shift",
                    "message": f"Referral rate changed from {k2 / len(base):.0%} to {k1 / len(recent):.0%} in "
                    f"the last 7 days (p={p:.3f}). Check for a change in patient mix or equipment.",
                }
            )
    if summary.get("missed_positives_low_risk"):
        alerts.append(
            {
                "level": "critical",
                "code": "missed_positive",
                "message": f"{summary['missed_positives_low_risk']} confirmed-positive patient(s) were called Low "
                "risk. Review these cases and consider re-onboarding.",
            }
        )
    return alerts
