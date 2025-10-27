import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from ..robodog.dog_bot_brain import RoboDogBrain
from ..utils.logging import get_logger

log = get_logger("API")
CFG = os.getenv("VCT_CONFIG", "vct/config.yaml")
SIM = os.getenv("VCT_SIMULATE", "1") == "1"
GPIO_PIN = int(os.getenv("VCT_GPIO_PIN", "0")) or None
brain = RoboDogBrain(cfg_path=CFG, gpio_pin=GPIO_PIN, simulate=SIM)

API_KEY = os.getenv("VCT_API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
ENFORCE_HTTPS = os.getenv("VCT_ENFORCE_HTTPS", "0") == "1"

app = FastAPI(title="VCT API", version="0.14.0")


def _assert_api_key(api_key: str | None) -> None:
    if API_KEY is None:
        log.warning(
            "API key authentication disabled because VCT_API_KEY is not set. "
            "Set VCT_API_KEY to enable authentication."
        )
        return
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    _assert_api_key(api_key)
    return api_key


@app.middleware("http")
async def enforce_https_middleware(request: Request, call_next):
    if ENFORCE_HTTPS:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        if proto != "https":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="HTTPS is required when VCT_ENFORCE_HTTPS=1",
            )
    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception during request processing", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"ok": False, "detail": "Internal server error"},
    )


class ActIn(BaseModel):
    text: str = Field(..., min_length=1)
    confidence: float = Field(0.85, ge=0.0, le=1.0)
    reward_bias: float = Field(0.5, ge=0.0, le=1.0)
    mood: float = Field(0.0, ge=-1.0, le=1.0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


@app.get("/health")
def health(api_key: str | None = Depends(verify_api_key)) -> dict[str, bool | str]:
    return {"status": "ok", "simulate": SIM}


@app.post("/robot/act")
def act(inp: ActIn, api_key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    try:
        out = brain.handle_command(inp.text, inp.confidence, inp.reward_bias, inp.mood)
    except Exception as exc:  # pragma: no cover - defensive programming
        log.exception("Failed to handle robot command", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute robot command",
        ) from exc
    return {"ok": True, "result": out}
