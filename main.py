"""
main.py — FastAPI entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.parse import router as parse_router

app = FastAPI(
    title="Data Pipeline API",
    description=(
        "Preprocessing microservice for retail/clothing analytics. "
        "Phase 1: /parse — column mapping & preview. "
        "Phase 2: /clean — data cleaning & schema enforcement (coming soon)."
    ),
    version="0.1.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Adjust origins to match your Next.js deployment URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(parse_router)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": app.version}