"""
routers/clean.py

POST /clean
  - Accepts: multipart/form-data
      file    : UploadFile (.csv / .xlsx / .xls)
      mapping : JSON string (ConfirmedMapping)
  - Returns:
      200 CleanResponse  — cleaning complete, entities ready for DB insertion
      202 CleanResponse  — action_required: user must decide on missing ML fields
      400 / 413 / 415 / 422 / 500 on errors
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.clean_models import CleanResponse, ConfirmedMapping
from app.services.cleaner import clean_file

router = APIRouter(prefix="/clean", tags=["Clean"])

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB   = 50


@router.post(
    "",
    response_model=CleanResponse,
    summary="Clean and normalize an uploaded file using the confirmed column mapping",
    response_description=(
        "DB-ready entity records grouped by table, "
        "with ML-derived fields, failed row report, and warnings. "
        "Returns 202 if user action is required for missing ML fields."
    ),
    responses={
        200: {"description": "Cleaning complete — entities ready for DB insertion"},
        202: {"description": "Action required — some ML fields need user decisions"},
        400: {"description": "Empty file or invalid mapping JSON"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "File could not be parsed or cleaned"},
        500: {"description": "Unexpected server error"},
    },
)
async def clean_upload(
    file: UploadFile = File(
        ...,
        description="CSV or Excel file (.csv / .xlsx / .xls) — same file as sent to /parse",
    ),
    mapping: str = Form(
        ...,
        description=(
            "JSON string of ConfirmedMapping — the user-reviewed and confirmed "
            "column mapping from /parse output."
        ),
    ),
) -> JSONResponse:

    # ── Validate filename ─────────────────────────────────────────────────────
    filename = file.filename or ""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # ── Read bytes ────────────────────────────────────────────────────────────
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Parse confirmed mapping ───────────────────────────────────────────────
    try:
        mapping_dict  = json.loads(mapping)
        confirmed_map = ConfirmedMapping(**mapping_dict)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mapping JSON: {str(exc)}",
        ) from exc

    # ── Run cleaning pipeline ─────────────────────────────────────────────────
    try:
        result = await clean_file(file_bytes, filename, confirmed_map)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cleaning pipeline failed: {str(exc)}",
        ) from exc

    # ── 202 if action required ────────────────────────────────────────────────
    status_code = 202 if result.action_required else 200

    return JSONResponse(
        content=result.model_dump(),
        status_code=status_code,
    )