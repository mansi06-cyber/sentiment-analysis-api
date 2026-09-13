"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.routes.analyze import router as analyze_router
from app.routes.health import router as health_router
from app.services import sentiment_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model on startup."""
    logger.info("Initializing application and pre-loading sentiment model...")
    try:
        sentiment_service.load_model()
    except Exception as exc:
        logger.error(f"Failed to load model during startup: {exc}")
    yield
    logger.info("Shutting down application...")


# FastAPI application instance
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api prefix
app.include_router(health_router, prefix="/api")
app.include_router(analyze_router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint welcoming users and providing documentation links."""
    return {
        "message": f"Welcome to the {API_TITLE} v{API_VERSION}",
        "documentation": "/docs",
        "health_check": "/api/health",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "batch_analyze": "POST /api/analyze/batch",
            "health": "GET /api/health"
        }
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom exception handler for Pydantic validation errors for clean, readable feedback."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        message = error.get("msg", "Invalid input")
        errors.append({
            "field": field,
            "message": message,
            "type": error.get("type")
        })

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": errors
        }
    )
