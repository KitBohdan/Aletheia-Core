import os
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, constr
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ..robodog.dog_bot_brain import RoboDogBrain
from ..utils.logging import get_logger
from ..utils.metrics import (
    observe_command_latency,
    record_api_request,
    record_command,
)
from .security import APIKeyAuthError, require_api_key

log = get_logger("API")
CFG = os.getenv("VCT_CONFIG", "vct/config.yaml")
SIM = os.getenv("VCT_SIMULATE", "1") == "1"
GPIO_PIN = int(os.getenv("VCT_GPIO_PIN", "0")) or None
_https_env = os.getenv("VCT_REQUIRE_HTTPS", "0")
REQUIRE_HTTPS = _https_env == "1"
brain = RoboDogBrain(cfg_path=CFG, gpio_pin=GPIO_PIN, simulate=SIM)

ACT_ENDPOINT = "/robot/act"

app = FastAPI(title="VCT API", version="0.14.0")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        method = request.method
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = perf_counter() - start
            record_api_request(path, method, 500)
            if path == ACT_ENDPOINT:
                observe_command_latency(path, duration)
            raise
        duration = perf_counter() - start
        record_api_request(path, method, response.status_code)
        if path == ACT_ENDPOINT:
            observe_command_latency(path, duration)
        return response


app.add_middleware(MetricsMiddleware)
if REQUIRE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)


@app.exception_handler(APIKeyAuthError)
async def api_key_auth_exception_handler(_: Request, exc: APIKeyAuthError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
        )

    log.exception("Unhandled exception in API request")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "simulate": SIM}


class ActIn(BaseModel):
    text: constr(min_length=1, strip_whitespace=True)  # type: ignore[valid-type]
    confidence: float = Field(0.85, ge=0.0, le=1.0)
    reward_bias: float = Field(0.5, ge=0.0, le=1.0)
    mood: float = Field(0.0, ge=-1.0, le=1.0)


@app.post("/robot/act")
def act(inp: ActIn, _: str = Depends(require_api_key)) -> dict[str, Any]:
    record_command("api")
    out = brain.handle_command(inp.text, inp.confidence, inp.reward_bias, inp.mood)
    return {"ok": True, "result": out}


@app.get("/metrics")
def metrics(_: str = Depends(require_api_key)) -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
