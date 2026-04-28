"""
app/services/segmentation/common/job_tracker.py

Generic CRUD helpers for the analysis_job table.
Reusable across product_segmentation, future customer_segmentation, etc.
"""
from __future__ import annotations

import uuid
import json
from datetime import datetime
from typing import Any

from app.core.database import get_db_connection


def _log(msg: str) -> None:
    print(f"[job_tracker] {msg}", flush=True)


def create_job(business_id: str, job_type: str, triggered_by: str) -> str:
    """
    Insert a new job in 'queued' status. Returns the job_id (uuid string).
    """
    job_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO analysis_job
                   (id, business_id, type, status, progress, triggered_by)
                   VALUES (%s, %s, %s, 'queued', 0, %s)""",
                (job_id, business_id, job_type, triggered_by),
            )
        conn.commit()
    finally:
        conn.close()
    _log(f"created job {job_id} type={job_type} business={business_id} trigger={triggered_by}")
    return job_id


def update_job(
    job_id:      str,
    status:      str | None = None,
    progress:    int | None = None,
    detail:      str | None = None,
    error:       str | None = None,
    result_meta: dict[str, Any] | None = None,
    started:     bool = False,
    finished:    bool = False,
) -> None:
    """
    Patch any subset of job fields. Skips unsent fields.
    """
    sets: list[str] = []
    args: list[Any] = []

    if status is not None:
        sets.append("status = %s"); args.append(status)
    if progress is not None:
        sets.append("progress = %s"); args.append(progress)
    if detail is not None:
        sets.append("detail = %s"); args.append(detail)
    if error is not None:
        sets.append("error = %s"); args.append(error)
    if result_meta is not None:
        sets.append("result_meta = %s::jsonb"); args.append(json.dumps(result_meta))
    if started:
        sets.append("started_at = NOW()")
    if finished:
        sets.append("finished_at = NOW()")

    if not sets:
        return

    args.append(job_id)
    sql = f"UPDATE analysis_job SET {', '.join(sets)} WHERE id = %s"

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
        conn.commit()
    finally:
        conn.close()


def mark_skipped(job_id: str, reason: str) -> None:
    update_job(job_id, status="skipped", detail=reason, finished=True, progress=100)


def mark_failed(job_id: str, error: str) -> None:
    update_job(job_id, status="failed", error=error, finished=True)


def mark_done(job_id: str, result_meta: dict[str, Any], detail: str = "Complete") -> None:
    update_job(
        job_id,
        status="done",
        progress=100,
        detail=detail,
        result_meta=result_meta,
        finished=True,
    )


def get_active_jobs(business_id: str) -> list[dict[str, Any]]:
    """Return jobs currently running or queued for a business."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, type, status, progress, detail, started_at, created_at
                   FROM analysis_job
                   WHERE business_id = %s
                     AND status IN ('queued', 'running')
                   ORDER BY created_at DESC""",
                (business_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def get_job(job_id: str) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, business_id, type, status, progress, detail, error,
                          result_meta, triggered_by, started_at, finished_at, created_at
                   FROM analysis_job WHERE id = %s""",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    finally:
        conn.close()