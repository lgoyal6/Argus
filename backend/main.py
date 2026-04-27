from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import runs, metrics, decisions

app = FastAPI(title="AutoDebug API")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://3.137.141.220:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── routers ────────────────────────────────────────────────────────────────────
app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(metrics.router, prefix="/runs", tags=["metrics"])
app.include_router(decisions.router, prefix="/runs", tags=["decisions"])


# ── health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "AutoDebug API"}
