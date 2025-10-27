from fastapi import FastAPI
from pydantic import BaseModel
from ..robodog.dog_bot_brain import RoboDogBrain
from ..utils.logging import get_logger
import os

log = get_logger("API")
CFG = os.getenv("VCT_CONFIG", "vct/config.yaml")
SIM = os.getenv("VCT_SIMULATE", "1") == "1"
GPIO_PIN = int(os.getenv("VCT_GPIO_PIN", "0")) or None
brain = RoboDogBrain(cfg_path=CFG, gpio_pin=GPIO_PIN, simulate=SIM)

app = FastAPI(title="VCT API", version="0.14.0")

@app.get("/health")
def health(): return {"status": "ok", "simulate": SIM}

class ActIn(BaseModel):
    text: str
    confidence: float = 0.85
    reward_bias: float = 0.5
    mood: float = 0.0

@app.post("/robot/act")
def act(inp: ActIn):
    out = brain.handle_command(inp.text, inp.confidence, inp.reward_bias, inp.mood)
    return {"ok": True, "result": out}