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
from app.services.ml_exporter import export_ml_dataset
import asyncio

router = APIRouter(prefix="/clean", tags=["Clean"])


class CleanRequest(BaseModel):
    file_id:     str
    mapping:     ConfirmedMapping
    business_id: str

async def _background_maybe_segment(business_id: str) -> None:
    """
    After a successful import, evaluate whether product segmentation
    should be re-run. The trigger checks thresholds; if it passes,
    a new job is created and runs in the background.
    """
    try:
        from app.services.segmentation.product.triggers import ProductSegmentationTrigger
        from app.services.segmentation.common.job_tracker import create_job
        from app.services.segmentation.product.runner import run_product_segmentation_job

        decision = ProductSegmentationTrigger().evaluate(business_id)
        print(f"[clean→segment] decision: {decision.reason} — {decision.detail}", flush=True)

        if not decision.should_run:
            return

        job_id = create_job(business_id, "product_segmentation", "auto:clean")
        asyncio.ensure_future(run_product_segmentation_job(business_id, job_id))
    except Exception as exc:
        print(f"[clean→segment] background trigger failed (non-critical): {exc}", flush=True)

async def _background_export(
    business_id: str,
    customers:   list[dict],
    products:    list[dict],
    orders:      list[dict],
    order_items: list[dict],
) -> None:
    """
    Fire-and-forget ML export.
    Called after the SSE stream closes — never blocks the user.
    """
    try:
        result = await export_ml_dataset(
            business_id=business_id,
            customers  =customers,
            products   =products,
            orders     =orders,
            order_items=order_items,
        )
        print(f"[ml_export] background export done: {result['file_id']} "
              f"({result['row_count']:,} rows, {result['size_bytes']/1024/1024:.2f} MB)", flush=True)
    except Exception as exc:
        print(f"[ml_export] background export failed (non-critical): {exc}", flush=True)

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

        e = clean_result.entities
        customers_d   = [c.model_dump() for c in e.customers]
        products_d    = [p.model_dump() for p in e.products]
        orders_d      = [o.model_dump() for o in e.orders]
        order_items_d = [i.model_dump() for i in e.order_items]

        try:
            stats = store_entities(
                business_id=body.business_id,
                customers  =customers_d,
                products   =products_d,
                orders     =orders_d,
                order_items=order_items_d,
            )
        except Exception as exc:
            error_msg = str(exc).replace('"', "'")
            yield f'data: {{"__final__":true,"error":"DB write failed: {error_msg}"}}\n\n'
            return

        # Final event — stream ends here, user is unblocked immediately
        final = {
            "__final__": True,
            "summary":   clean_result.summary.model_dump(),
            "stats":     stats,
            "warnings":  clean_result.warnings,
        }
        yield f"data: {json.dumps(final)}\n\n"

        # ── Fire-and-forget product segmentation trigger ─────────────────
        asyncio.ensure_future(_background_maybe_segment(body.business_id))

        # ── Fire-and-forget ML export ─────────────────────────────────────
        # Runs after the SSE connection closes. User never waits for this.
        asyncio.ensure_future(
            _background_export(
                business_id=body.business_id,
                customers  =customers_d,
                products   =products_d,
                orders     =orders_d,
                order_items=order_items_d,
            )
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )