"""
routers/parse.py

POST /parse
  Accepts: JSON body { file_id: str }
  Downloads the file from Supabase Storage using the file_id.
  Returns: ParseResponse (mapping + confidence + sample)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.storage import download_file
from app.models.parse_models import ParseResponse
from app.services.parser import parse_file

router = APIRouter(prefix="/parse", tags=["Parse"])


class ParseRequest(BaseModel):
    file_id: str


@router.post(
    "",
    response_model=ParseResponse,
    summary="Parse a file stored in Supabase Storage",
    response_description=(
        "Detected column mapping with confidence scores, "
        "ML coverage report, attribute columns, unmapped columns, "
        "and a 5-row sample."
    ),
)
async def parse_upload(body: ParseRequest) -> ParseResponse:
    if not body.file_id:
        raise HTTPException(status_code=400, detail="file_id is required.")

    # ── Fetch file from Supabase Storage ─────────────────────────────────────
    try:
        file_bytes, filename = await download_file(body.file_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch file from storage: {str(exc)}",
        ) from exc

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File in storage is empty.")

    # ── Parse ─────────────────────────────────────────────────────────────────
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