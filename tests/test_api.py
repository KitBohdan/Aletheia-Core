import os

os.environ.setdefault("VCT_API_KEY", "test-api-key")

from fastapi.testclient import TestClient

from vct.api.app import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_act_endpoint():
    c = TestClient(app)
    r = c.post(
        "/robot/act",
        json={"text": "сидіти", "confidence": 0.9},
        headers={"X-API-Key": os.environ["VCT_API_KEY"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["result"]["action"] in ("SIT", "NONE")
