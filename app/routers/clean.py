"""
POST /clean
  Accepts: JSON { file_id, mapping, business_id }
  1. Downloads file from Supabase Storage
  2. Streams SSE progress through clean_stream
  3. Writes entities directly to Neon from Railway (bypasses Vercel 10s limit)
  4. Final SSE event: { __final__, summary, stats, warnings } — NO entity blob
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.storage import download_file
from app.core.database import store_entities
from app.models.clean_models import CleanResponse, ConfirmedMapping, ProgressEvent
from app.services.cleaner import clean_stream

router = APIRouter(prefix="/clean", tags=["Clean"])


class CleanRequest(BaseModel):
    file_id:     str
    mapping:     ConfirmedMapping
    business_id: str


@router.post("", summary="Clean + store — streams SSE progress, writes DB on Railway")
async def clean_upload(body: CleanRequest) -> StreamingResponse:
    if not body.file_id:
        raise HTTPException(status_code=400, detail="file_id is required.")
    if not body.business_id:
        raise HTTPException(status_code=400, detail="business_id is required.")

    try:
        file_bytes, filename = await download_file(body.file_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage fetch failed: {exc}") from exc

    if not file_bytes:
        raise HTTPException(status_code=400, detail="File in storage is empty.")

    async def event_generator():
        clean_result: CleanResponse | None = None

        # Stream all progress events; capture the final CleanResponse
        async for event in clean_stream(file_bytes, filename, body.mapping):
            if isinstance(event, ProgressEvent):
                yield f"data: {json.dumps(event.model_dump())}\n\n"
            elif isinstance(event, CleanResponse):
                clean_result = event

        if clean_result is None:
            yield 'data: {"__final__":true,"error":"Cleaner produced no result"}\n\n'
            return

        # ── Write to Neon from Railway (no Vercel timeout) ────────────────
        yield (
            'data: {"stage":"storing","pct":96,'
            '"detail":"Saving to your database…","counts":{}}\n\n'
        )

        try:
            e = clean_result.entities
            stats = store_entities(
                business_id=body.business_id,
                customers  =[c.model_dump() for c in e.customers],
                products   =[p.model_dump() for p in e.products],
                orders     =[o.model_dump() for o in e.orders],
                order_items=[i.model_dump() for i in e.order_items],
            )
        except Exception as exc:
            error_msg = str(exc).replace('"', "'")
            yield f'data: {{"__final__":true,"error":"DB write failed: {error_msg}"}}\n\n'
            return

        # Final event — small payload, no entity blobs
        final = {
            "__final__": True,
            "summary":   clean_result.summary.model_dump(),
            "stats":     stats,
            "warnings":  clean_result.warnings,
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )