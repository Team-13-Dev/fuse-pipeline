"""
main.py — FastAPI entry point
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.parse import router as parse_router
from app.routers.clean import router as clean_router

app = FastAPI(
    title="FUSE Data Pipeline",
    description=(
        "Preprocessing microservice for retail/clothing analytics. "
        "Phase 1 — /parse: column mapping & preview. "
        "Phase 2 — /clean: data cleaning, normalization & DB-ready entity assembly."
    ),
    version="0.2.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to Next.js URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(parse_router)
app.include_router(clean_router)

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": app.version}