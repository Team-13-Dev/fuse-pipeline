from fastapi import FastAPI

app = FastAPI(
    title="FUSE Pipeline",
    description="Data ingestion and normalization microservice for FUSE CRM",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "fuse-pipeline",
        "version": "0.1.0",
    }