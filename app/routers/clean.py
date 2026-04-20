"""
routers/clean.py  — v2

POST /clean
  Accepts: JSON { file_id, mapping }
  Streams SSE progress events, then final CleanResponse as the last event.

SSE format:
  data: {"stage":"loading","pct":2,"detail":"Reading your file...","counts":{}}
  data: {"stage":"done","pct":100,"detail":"All done.","counts":{...}}
  data: {"__final__": true, "summary": {...}, "entities": {...}, ...}
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.storage import download_file
from app.models.clean_models import CleanResponse, ConfirmedMapping, ProgressEvent
from app.services.cleaner import clean_stream

router = APIRouter(prefix="/clean", tags=["Clean"])


class CleanRequest(BaseModel):
    file_id: str
    mapping: ConfirmedMapping


@router.post(
    "",
    summary="Clean a file from Supabase Storage — streams SSE progress",
)
async def clean_upload(body: CleanRequest) -> StreamingResponse:
    if not body.file_id:
        raise HTTPException(status_code=400, detail="file_id is required.")

    try:
        file_bytes, filename = await download_file(body.file_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage fetch failed: {exc}") from exc

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File in storage is empty.")

    async def event_generator():
        async for event in clean_stream(file_bytes, filename, body.mapping):
            if isinstance(event, ProgressEvent):
                data = json.dumps(event.model_dump())
                yield f"data: {data}\n\n"
            elif isinstance(event, CleanResponse):
                # Final result — marked with __final__ so the client knows to stop
                payload = event.model_dump()
                payload["__final__"] = True
                yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":                "no-cache",
            "X-Accel-Buffering":            "no",   # disable nginx buffering
            "Access-Control-Allow-Origin":  "*",
        },
    )