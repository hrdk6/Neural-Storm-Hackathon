"""FastAPI service.

GET  /healthz                         liveness
GET  /readyz                          model bundle loaded and DB reachable
GET  /metrics                         Prometheus metrics
GET  /v1/info                         bundle, tasks, intended use
POST /v1/screen                       screen one image (multipart)
POST /v1/screenings/{id}/outcome      record the confirmatory test result (for monitoring)
GET  /v1/sites  POST /v1/sites        list / create sites
GET  /v1/sites/{id}
POST /v1/sites/{id}/onboard           upload onboarding pack (zip) -> background job
GET  /v1/jobs/{job_id}
GET  /v1/sites/{id}/report[.html]     site report card (active or ?version=)
POST /v1/sites/{id}/calibrations/{task}/{version}/activate   roll back / forward
GET  /v1/sites/{id}/monitoring        decision mix, gate/OOD rates, observed sens/spec
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import secrets
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from clinictune import DISCLAIMER, __version__, i18n
from clinictune.api.store import Store, jsonable
from clinictune.config import Settings, get_settings
from clinictune.imaging import ImageDecodeError
from clinictune.model.bundle import Bundle
from clinictune.site import report_card
from clinictune.site.ingest import PackError, read_pack, run_onboarding
from clinictune.triage import Screener

log = logging.getLogger("clinictune")

# Explicit types: the OS registry (notably on Windows) can map .js to text/plain, which `nosniff` would block.
for _ext, _type in (
    (".js", "text/javascript"),
    (".css", "text/css"),
    (".svg", "image/svg+xml"),
    (".woff2", "font/woff2"),
    (".webmanifest", "application/manifest+json"),
):
    mimetypes.add_type(_type, _ext)

SCREENS = Counter("clinictune_screenings_total", "Screenings", ["task", "decision", "reason"])
SCREEN_LATENCY = Histogram(
    "clinictune_screen_seconds", "Screening latency", ["task"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
)
ONBOARDINGS = Counter("clinictune_onboardings_total", "Onboarding jobs", ["task", "status"])
SITE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        d.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def setup_logging(s: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if s.log_json else logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(s.log_level)


class SiteIn(BaseModel):
    site_id: str = Field(..., description="lower-case id, e.g. 'phc-hosakote'")
    name: str = Field(..., max_length=200)


class OutcomeIn(BaseModel):
    outcome: int = Field(..., ge=0, le=1, description="confirmatory test result: 1 positive, 0 negative")


def _web_dir() -> Path | None:
    repo = Path(__file__).resolve().parents[3] / "web"
    if (repo / "index.html").exists():
        return repo
    pkg = resources.files("clinictune") / "_web"
    return Path(str(pkg)) if pkg.is_dir() else None


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while chunk := await upload.read(1 << 20):
        data += chunk
        if len(data) > max_bytes:
            raise HTTPException(413, f"upload exceeds {max_bytes // (1 << 20)} MB")
    return bytes(data)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s)
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        state["store"] = Store(s.db_path)
        state["store"].fail_stale_jobs()
        try:
            bundle = Bundle.load(s.bundle_dir, threads=s.ort_threads)
            state["bundle"] = bundle
            state["screener"] = Screener(bundle)
            log.info("loaded bundle %s from %s (demo=%s)", bundle.bundle_id, s.bundle_dir, bundle.is_demo)
        except Exception:
            log.exception("could not load model bundle from %s - /readyz will fail", s.bundle_dir)
        state["pool"] = ThreadPoolExecutor(max_workers=s.onboard_workers, thread_name_prefix="onboard")
        yield
        state["pool"].shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="ClinicTune",
        version=__version__,
        lifespan=lifespan,
        description="Site-adaptive screening aid. " + DISCLAIMER,
    )
    if s.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["x-frame-options"] = "DENY"
        if not request.url.path.startswith("/v1") or request.url.path.endswith(".html"):
            # The UI only runs its own scripts and only talks to this server. Inline styles are allowed for the
            # srcdoc report card and positioning; no inline or third-party scripts anywhere.
            response.headers.setdefault(
                "content-security-policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "font-src 'self'; connect-src 'self'; worker-src 'self'; frame-src 'self'; object-src 'none'; "
                "base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
            )
        # UI assets revalidate on every load (ETag) so an upgrade never runs stale JS; API results are never cached.
        response.headers.setdefault(
            "cache-control", "no-cache" if not request.url.path.startswith("/v1") else "no-store"
        )
        if request.url.path not in ("/healthz", "/metrics"):
            log.info(
                "request",
                extra={
                    "extra_fields": {
                        "request_id": rid,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                },
            )
        return response

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if s.api_key and not (x_api_key and secrets.compare_digest(x_api_key, s.api_key)):
            raise HTTPException(401, "missing or invalid X-API-Key")

    def store() -> Store:
        return state["store"]

    def screener() -> Screener:
        if "screener" not in state:
            raise HTTPException(503, "model bundle not loaded")
        return state["screener"]

    def check_site(site_id: str) -> None:
        if store().get_site(site_id) is None:
            raise HTTPException(404, f"site {site_id!r} not found")

    def check_task(task: str) -> None:
        if task not in screener().bundle.tasks:
            raise HTTPException(422, f"unknown task {task!r}; available: {sorted(screener().bundle.tasks)}")

    # ------------------------------------------------------------------ ops
    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        ok = "screener" in state
        try:
            store().list_sites()
        except Exception:  # pragma: no cover
            ok = False
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    v1 = [Depends(require_key)]

    @app.get("/v1/info", dependencies=v1)
    def info() -> dict[str, Any]:
        b = screener().bundle
        return jsonable(
            {
                "version": __version__,
                "bundle_id": b.bundle_id,
                "demo": b.is_demo,
                "backbones": {name: bb.spec.to_dict() for name, bb in b.backbones.items()},
                "tasks": {
                    k: {
                        **v,
                        "target_sensitivity": b.tasks[k].target_sensitivity,
                        "internal_metrics": b.tasks[k].metrics.get("overall"),
                        "training": {kk: b.tasks[k].training.get(kk) for kk in ("sites", "datasets", "n")},
                        "gradability": b.tasks[k].gradability.metrics if b.tasks[k].gradability else None,
                        "similar_cases": b.tasks[k].library is not None,
                    }
                    for k, v in b.manifest["tasks"].items()
                },
                "datasets": b.manifest.get("datasets"),
                "languages": {k: {"translation_reviewed": v["reviewed"]} for k, v in i18n.LANGS.items()},
                "disclaimer": DISCLAIMER,
            }
        )

    # ------------------------------------------------------------------ screening
    @app.post("/v1/screen", dependencies=v1)
    async def screen(
        request: Request,
        file: Annotated[UploadFile, File()],
        task: Annotated[str, Form()],
        site_id: Annotated[str | None, Form()] = None,
        lang: Annotated[str, Form()] = "en",
        heatmap: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        check_task(task)
        site_cal = None
        if site_id:
            check_site(site_id)
            site_cal = store().active_calibration(site_id, task)
        data = await _read_limited(file, int(s.max_upload_mb * (1 << 20)))
        t0 = time.perf_counter()
        try:
            # ONNX inference is CPU-bound: run in the threadpool, not on the event loop.
            from starlette.concurrency import run_in_threadpool

            result = await run_in_threadpool(screener().screen, data, task, site=site_cal, lang=lang, heatmap=heatmap)
        except ImageDecodeError as exc:
            raise HTTPException(422, str(exc)) from exc
        SCREEN_LATENCY.labels(task).observe(time.perf_counter() - t0)
        SCREENS.labels(task, str(result["decision"]), result.get("reason") or "").inc()
        rid = request.state.request_id
        result["request_id"] = rid
        if s.store_audit_log:
            store().log_screening(rid, site_id, result)
        return jsonable(result)

    @app.post("/v1/screenings/{request_id}/outcome", dependencies=v1)
    def outcome(request_id: str, body: OutcomeIn) -> dict[str, Any]:
        if not store().record_outcome(request_id, body.outcome):
            raise HTTPException(404, "screening not found")
        return {"request_id": request_id, "outcome": body.outcome}

    # ------------------------------------------------------------------ sites
    @app.get("/v1/sites", dependencies=v1)
    def list_sites() -> list[dict[str, Any]]:
        return store().list_sites()

    @app.post("/v1/sites", status_code=201, dependencies=v1)
    def create_site(body: SiteIn) -> dict[str, Any]:
        if not SITE_ID.match(body.site_id):
            raise HTTPException(422, "site_id must be 2-64 chars: lower-case letters, digits, '-' or '_'")
        if store().get_site(body.site_id):
            raise HTTPException(409, "site already exists")
        return store().create_site(body.site_id, body.name)

    @app.get("/v1/sites/{site_id}", dependencies=v1)
    def get_site(site_id: str) -> dict[str, Any]:
        check_site(site_id)
        return store().get_site(site_id)  # type: ignore[return-value]

    @app.post("/v1/sites/{site_id}/onboard", status_code=202, dependencies=v1)
    async def onboard(
        site_id: str,
        task: Annotated[str, Form()],
        pack: Annotated[UploadFile, File(description="zip: images + labels.csv (filename,label[,patient_id])")],
    ) -> dict[str, Any]:
        check_site(site_id)
        check_task(task)
        data = await _read_limited(pack, int(s.max_onboard_zip_mb * (1 << 20)))
        try:
            items = read_pack(data, s.max_onboard_images, int(s.max_onboard_zip_mb * 4 * (1 << 20)))
        except PackError as exc:
            raise HTTPException(422, str(exc)) from exc
        st = store()
        job_id = st.create_job(site_id, task)
        bundle = screener().bundle

        def work() -> None:
            st.update_job(job_id, status="running", message="starting")
            try:
                calib, report = run_onboarding(
                    bundle,
                    site_id,
                    task,
                    items,
                    progress=lambda p, m: st.update_job(job_id, progress=round(p, 3), message=m),
                )
                version = st.add_calibration(calib, report, activate=True)
                st.update_job(
                    job_id,
                    status="done",
                    progress=1.0,
                    message="done",
                    result={"version": version, "verdict": report["verdict"]},
                )
                ONBOARDINGS.labels(task, "done").inc()
            except (PackError, ValueError) as exc:
                st.update_job(job_id, status="failed", message=str(exc))
                ONBOARDINGS.labels(task, "failed").inc()
            except Exception as exc:  # pragma: no cover
                log.exception("onboarding job %s failed", job_id)
                st.update_job(job_id, status="failed", message=f"internal error: {type(exc).__name__}")
                ONBOARDINGS.labels(task, "failed").inc()

        state["pool"].submit(work)
        return {"job_id": job_id, "status": "queued", "images": len(items)}

    @app.get("/v1/jobs/{job_id}", dependencies=v1)
    def job(job_id: str) -> dict[str, Any]:
        j = store().get_job(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j

    def _report(site_id: str, task: str, version: int | None) -> dict[str, Any]:
        check_site(site_id)
        r = store().report(site_id, task, version)
        if r is None:
            raise HTTPException(404, "no calibration for this site/task yet - onboard first")
        return r

    @app.get("/v1/sites/{site_id}/report", dependencies=v1)
    def report(site_id: str, task: str, version: int | None = None) -> dict[str, Any]:
        return _report(site_id, task, version)

    @app.get("/v1/sites/{site_id}/report.html", response_class=HTMLResponse, dependencies=v1)
    def report_html(site_id: str, task: str, version: int | None = None) -> str:
        return report_card.render(_report(site_id, task, version))

    @app.post("/v1/sites/{site_id}/calibrations/{task}/{version}/activate", dependencies=v1)
    def activate(site_id: str, task: str, version: int) -> dict[str, Any]:
        check_site(site_id)
        if not store().activate(site_id, task, version):
            raise HTTPException(404, "calibration version not found")
        return {"site_id": site_id, "task": task, "active_version": version}

    @app.get("/v1/sites/{site_id}/monitoring", dependencies=v1)
    def monitoring(site_id: str, task: str, days: Annotated[int, Query(ge=1, le=365)] = 30) -> dict[str, Any]:
        check_site(site_id)
        return store().monitoring(site_id, task, days)

    @app.get("/v1/sites/{site_id}/screenings.csv", dependencies=v1)
    def screenings_csv(site_id: str, days: Annotated[int, Query(ge=1, le=3650)] = 90) -> Response:
        """Screening log for district reporting: decisions, outcomes and versions - no images, no patient data."""
        check_site(site_id)
        body = store().export_csv(site_id, days)
        return Response(
            body,
            media_type="text/csv",
            headers={"content-disposition": f'attachment; filename="{site_id}-screenings.csv"'},
        )

    @app.get("/v1/evidence", dependencies=v1)
    def evidence() -> dict[str, Any]:
        """Validation evidence shipped with the bundle (LOSO, shortcut, subgroup, backbone benchmarks)."""
        ev = screener().bundle.evidence()
        if ev is None:
            raise HTTPException(404, "this bundle has no evidence.json - run `clinictune evidence`")
        return jsonable(ev)

    @app.get("/v1/library/{task}/{name}", include_in_schema=False)
    def library_thumb(task: str, name: str) -> Response:
        """Reference-case thumbnails for similar-case retrieval (served from the read-only bundle)."""
        if not re.fullmatch(r"\d{4}\.jpg", name) or task not in screener().bundle.tasks:
            raise HTTPException(404, "not found")
        path = screener().bundle.root / "library" / task / name
        if not path.exists():
            raise HTTPException(404, "not found")
        return Response(path.read_bytes(), media_type="image/jpeg", headers={"cache-control": "max-age=86400"})

    web = _web_dir()
    if web is not None and (web / "icon.svg").exists():
        icon_bytes = (web / "icon.svg").read_bytes()

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> Response:  # browsers ask for this path whatever the page declares
            return Response(icon_bytes, media_type="image/svg+xml", headers={"cache-control": "max-age=86400"})

    if web is not None:
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def app_factory() -> FastAPI:
    return create_app()
