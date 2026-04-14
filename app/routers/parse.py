"""
routers/parse.py

POST /parse
  - Accepts: multipart/form-data with a single file field named "file"
  - Supports: .csv, .xlsx, .xls
  - Returns: ParseResponse (mapping + confidence + sample — NO data transformation)
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.parse_models import ParseResponse
from app.services.parser import parse_file

router = APIRouter(prefix="/parse", tags=["Parse"])

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 50


@router.post(
    "",
    response_model=ParseResponse,
    summary="Upload a CSV or Excel file and get column mapping + preview",
    response_description=(
        "Detected column mapping with confidence scores, "
        "ML coverage report, clothing attribute columns, "
        "unmapped columns, and a 5-row sample."
    ),
)
async def parse_upload(
    file: UploadFile = File(..., description="CSV or Excel file (.csv / .xlsx / .xls)"),
) -> ParseResponse:
    # ── Validate filename ────────────────────────────────────────────────────
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # ── Read bytes ───────────────────────────────────────────────────────────
    file_bytes = await file.read()

    # ── Size guard ───────────────────────────────────────────────────────────
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        result = await parse_file(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse file: {str(exc)}",
        ) from exc

    return result