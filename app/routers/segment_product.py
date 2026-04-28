"""
app/routers/segment_product.py

Routes:
  POST /segment/product/maybe   — checks thresholds, runs only if warranted
  POST /segment/product/force   — bypasses thresholds, always runs (still respects min-15 gate)

Both endpoints return immediately with a job_id. The work happens in the background.
Status polling: GET /jobs/{job_id} (see jobs.py).
"""

import asyncio
from fastapi import APIRouter, HTTPException

from app.services.segmentation.common.job_tracker import create_job, update_job
from app.services.segmentation.product.schemas import (
    MaybeSegmentRequest, ForceSegmentRequest, MaybeSegmentResponse,
)
from app.services.segmentation.product.triggers import ProductSegmentationTrigger
from app.services.segmentation.product.runner import run_product_segmentation_job


router = APIRouter(prefix="/segment/product", tags=["Segmentation:Product"])

JOB_TYPE = "product_segmentation"


@router.post(
    "/maybe",
    response_model=MaybeSegmentResponse,
    summary="Conditionally trigger product segmentation if thresholds are met",
)
async def maybe_segment(body: MaybeSegmentRequest) -> MaybeSegmentResponse:
    if not body.business_id:
        raise HTTPException(status_code=400, detail="business_id is required.")

    decision = ProductSegmentationTrigger().evaluate(body.business_id)

    if not decision.should_run:
        return MaybeSegmentResponse(
            job_id=None, will_run=False,
            reason=decision.reason, detail=decision.detail,
        )

    job_id = create_job(body.business_id, JOB_TYPE, body.triggered_by)
    asyncio.ensure_future(run_product_segmentation_job(body.business_id, job_id))

    return MaybeSegmentResponse(
        job_id=job_id, will_run=True,
        reason=decision.reason, detail=decision.detail,
    )


@router.post(
    "/force",
    response_model=MaybeSegmentResponse,
    summary="Force a product segmentation run (still respects min-products gate)",
)
async def force_segment(body: ForceSegmentRequest) -> MaybeSegmentResponse:
    if not body.business_id:
        raise HTTPException(status_code=400, detail="business_id is required.")

    # Even forced runs respect the hard minimum-products gate
    decision = ProductSegmentationTrigger().evaluate(body.business_id)
    if decision.reason == "insufficient_products":
        return MaybeSegmentResponse(
            job_id=None, will_run=False,
            reason=decision.reason, detail=decision.detail,
        )

    job_id = create_job(body.business_id, JOB_TYPE, body.triggered_by)
    asyncio.ensure_future(run_product_segmentation_job(body.business_id, job_id))

    return MaybeSegmentResponse(
        job_id=job_id, will_run=True,
        reason="forced", detail="Manual refresh requested.",
    )