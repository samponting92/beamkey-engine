"""
BeamKey API.

POST /analyse                   upload a PDF -> members found on each sheet (JSON)
GET  /jobs/{id}/page/{n}.png    sheet image for the review screen
POST /jobs/{id}/render          send back the (edited) members -> colour-coded PDF
POST /jobs/{id}/report          optional: Claude writes a short report and flags conflicts
POST /process                   one shot: upload a PDF -> colour-coded PDF
GET  /health
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import engine

JOBS = Path(os.environ.get("BEAMKEY_JOBS_DIR", "/tmp/beamkey-jobs"))
JOBS.mkdir(parents=True, exist_ok=True)
JOB_TTL_HOURS = float(os.environ.get("BEAMKEY_JOB_TTL_HOURS", "24"))
MAX_MB = float(os.environ.get("BEAMKEY_MAX_MB", "60"))
API_KEY = os.environ.get("BEAMKEY_API_KEY", "")
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="BeamKey API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["Content-Disposition"])


def check_key(x_beamkey_key: str = Header(default="")):
    if API_KEY and x_beamkey_key != API_KEY:
        raise HTTPException(401, "Missing or wrong X-BeamKey-Key header")


def _cleanup():
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for d in JOBS.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


async def _read_pdf(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB:.0f} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF")
    return data


def _job(job_id: str) -> Path:
    d = JOBS / job_id
    if not d.exists() or "/" in job_id or ".." in job_id:
        raise HTTPException(404, "Job not found or expired")
    return d


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyse", dependencies=[Depends(check_key)])
async def analyse(file: UploadFile = File(...)):
    _cleanup()
    data = await _read_pdf(file)
    try:
        pages = engine.analyse(data)
    except Exception as e:  # damaged or encrypted PDF
        raise HTTPException(422, f"Could not read this PDF: {e}")
    job_id = uuid.uuid4().hex
    d = JOBS / job_id
    d.mkdir()
    (d / "input.pdf").write_bytes(data)
    (d / "name.txt").write_text(Path(file.filename or "drawings.pdf").stem)
    (d / "auto.json").write_text(json.dumps(pages))
    warnings = []
    for i, p in enumerate(pages):
        p["page"] = i
        p["image"] = f"/jobs/{job_id}/page/{i}.png"
        if not p.pop("vector"):
            warnings.append(f"Sheet {i + 1} looks like a scan. Automatic placement needs a vector PDF, "
                            "so add members by hand on this sheet.")
        for c in p.get("conflicts", []):
            warnings.append(f"Sheet {i + 1}: {c}")
        if p.get("ocr"):
            warnings.append(f"Sheet {i + 1} has no text layer, so its labels were read by OCR. "
                            "Check member references carefully.")
    return {"job_id": job_id, "pages": pages, "warnings": warnings}


@app.get("/jobs/{job_id}/page/{n}.png")
def page_png(job_id: str, n: int, dpi: int = 110):
    d = _job(job_id)
    dpi = max(50, min(dpi, 200))
    cache = d / f"p{n}_{dpi}.png"
    if not cache.exists():
        try:
            cache.write_bytes(engine.page_png((d / "input.pdf").read_bytes(), n, dpi))
        except IndexError:
            raise HTTPException(404, "No such sheet")
    return Response(cache.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/jobs/{job_id}/render", dependencies=[Depends(check_key)])
async def render(job_id: str, body: dict):
    d = _job(job_id)
    pages = body.get("pages")
    if pages is None:
        pages = json.loads((d / "auto.json").read_text())
    else:
        # keep the checked version: auto.json vs final.json is how the engine gets better
        (d / "final.json").write_text(json.dumps(pages))
    out = engine.render((d / "input.pdf").read_bytes(), pages)
    name = (d / "name.txt").read_text() + " - colour coded.pdf"
    return Response(out, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/jobs/{job_id}/report", dependencies=[Depends(check_key)])
async def report(job_id: str, body: dict | None = None):
    d = _job(job_id)
    pages = (body or {}).get("pages") or json.loads((d / "auto.json").read_text())
    from report import write_report
    return write_report((d / "input.pdf").read_bytes(), pages)


@app.post("/process", dependencies=[Depends(check_key)])
async def process(file: UploadFile = File(...)):
    data = await _read_pdf(file)
    pages = engine.analyse(data)
    out = engine.render(data, pages)
    name = Path(file.filename or "drawings.pdf").stem + " - colour coded.pdf"
    return Response(out, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
