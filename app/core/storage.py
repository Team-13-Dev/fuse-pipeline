"""
app/core/storage.py

Supabase Storage client for the fuse-pipeline FastAPI service.
Downloads files from the pipeline-imports bucket using the file_id
(storage path) returned by the Next.js /api/onboarding/upload route.
"""

from __future__ import annotations

import os
import httpx

SUPABASE_URL             = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
IMPORT_BUCKET            = "pipeline-imports"


async def download_file(file_id: str) -> tuple[bytes, str]:
    """
    Download a file from Supabase Storage by its storage path (file_id).

    Returns:
        (file_bytes, filename)  — raw bytes and the original filename
                                  extracted from the storage path.

    Raises:
        ValueError  — if the file cannot be fetched (404, auth error, etc.)
    """
    url = (
        f"{SUPABASE_URL}/storage/v1/object/"
        f"{IMPORT_BUCKET}/{file_id}"
    )

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, headers=headers)

    if response.status_code == 404:
        raise ValueError(f"File not found in storage: {file_id}")

    if response.status_code != 200:
        raise ValueError(
            f"Supabase Storage error {response.status_code}: {response.text[:200]}"
        )

    # Extract filename from storage path: "{businessId}/{timestamp}-{filename}"
    filename = file_id.split("/")[-1]
    # Strip the leading timestamp prefix: "1234567890-myfile.xlsx" → "myfile.xlsx"
    if "-" in filename:
        filename = "-".join(filename.split("-")[1:])

    return response.content, filename


def delete_file(file_id: str) -> None:
    """
    Synchronously delete a file from Supabase Storage.
    Called after successful processing to free storage space.
    Fire-and-forget — errors are logged but not raised.
    """
    import httpx as _httpx

    url = (
        f"{SUPABASE_URL}/storage/v1/object/"
        f"{IMPORT_BUCKET}/{file_id}"
    )
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }
    try:
        with _httpx.Client(timeout=10.0) as client:
            client.delete(url, headers=headers)
    except Exception as exc:
        print(f"[storage] Warning: could not delete {file_id}: {exc}")