# VCT RoboDog — Voice Canine Trainer + RoboDog Brain

LLM-first стек для голосового дресурування собак з GPIO-диспенсером винагород і етичними запобіжниками.

## Швидкий старт
```bash
python -m pip install -U pip
pip install -e .
python -m vct.cli --simulate --cmd "сидіти"
```

## REST API
```bash
uvicorn vct.api.app:app --reload --port 8000
# health: GET /health
# act:    POST /robot/act {"text":"сидіти","confidence":0.9}
```