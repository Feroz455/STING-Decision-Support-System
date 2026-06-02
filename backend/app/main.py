from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.api.router import api_router
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


# Rate limiter — IP bazlı
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="STING — Drug Repositioning DSS for Childhood Acute Leukemia",
    version="1.0.0",
    description=(
        "**STING**: Development of a Drug Repositioning Decision Support System "
        "for Childhood Acute Leukemia by Digital Twin-Oriented Deep Learning\n\n"
        "Funded by The Scientific and Technological Research Council of Türkiye (TÜBİTAK) "
        "under the 1001 Programme · Project No: 123E383"
    ),
    lifespan=lifespan,
    # Production'da /docs ve /redoc kapalı
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
