import os
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from ..robodog.dog_bot_brain import RoboDogBrain
from ..utils.logging import get_logger
from ..utils.metrics import (
    observe_command_latency,
    record_api_request,
    record_command,
)

log = get_logger("API")
CFG = os.getenv("VCT_CONFIG", "vct/config.yaml")
SIM = os.getenv("VCT_SIMULATE", "1") == "1"
GPIO_PIN = int(os.getenv("VCT_GPIO_PIN", "0")) or None
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


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "simulate": SIM}


class ActIn(BaseModel):
    text: str
    confidence: float = 0.85
    reward_bias: float = 0.5
    mood: float = 0.0


@app.post("/robot/act")
def act(inp: ActIn) -> dict[str, Any]:
    record_command("api")
    out = brain.handle_command(inp.text, inp.confidence, inp.reward_bias, inp.mood)
    return {"ok": True, "result": out}


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
