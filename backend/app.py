from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.health import router as health_router
from backend.routes.analysis import router as analysis_router
from backend.routes.forecast import router as forecast_router
from backend.routes.records import router as records_router
from backend.routes.alerts import router as alerts_router

from database.connection import init_database


app = FastAPI(
    title="ARJUN Cyber Defence API",
    description=(
        "World-model based proactive cyber defence API "
        "for network attack forecasting."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Initialize the ARJUN database when FastAPI starts.
    """

    init_database()


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(
    health_router,
    prefix="/api",
)

app.include_router(
    analysis_router,
    prefix="/api",
)

app.include_router(
    forecast_router,
    prefix="/api",
)

app.include_router(
    records_router,
    prefix="/api",
)

app.include_router(
    alerts_router,
    prefix="/api",
)


@app.get("/")
def root():
    return {
        "application": "ARJUN",
        "description": "Proactive Cyber Defence World Model",
        "status": "running",
        "api_version": "1.0.0",
    }