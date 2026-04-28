"""
app/routers/jobs.py

Generic job status polling for all segmentation/ML features.

Routes:
  GET /jobs/{job_id}           — single job status
  GET /jobs/active/{business_id} — list active (queued/running) jobs
"""

from fastapi import APIRouter, HTTPException

from app.services.segmentation.common.job_tracker import get_job, get_active_jobs


router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", summary="Get a single job's current status")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # ISO-format datetimes for JSON friendliness
    for key in ("started_at", "finished_at", "created_at"):
        if job.get(key):
            job[key] = job[key].isoformat()

    return job


@router.get("/active/{business_id}", summary="Get all active jobs for a business")
async def get_business_active_jobs(business_id: str):
    jobs = get_active_jobs(business_id)
    for j in jobs:
        for key in ("started_at", "created_at"):
            if j.get(key):
                j[key] = j[key].isoformat()
    return {"jobs": jobs}