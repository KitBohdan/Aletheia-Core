from vct.engines.stt import RuleBasedSTT, WhisperSTT
from vct.engines.tts import OpenAITTS, PrintTTS


def test_rulebased_stt_mapping():
    stt = RuleBasedSTT()
    assert stt.transcribe(wav_path="data/examples/commands/sydity.wav") == "сидіти"


def test_whisper_stt_transcribe_with_stub(tmp_path):
    captured = {}

    class DummyModel:
        def transcribe(self, wav_path, fp16):
            captured["path"] = wav_path
            captured["fp16"] = fp16
            return {"text": "  лежати  "}

    loader_calls = {"count": 0}

    def loader():
        loader_calls["count"] += 1
        return DummyModel()

    stt = WhisperSTT(loader=loader)
    assert stt.transcribe(wav_path="sample.wav") == "лежати"
    assert captured["path"] == "sample.wav"
    assert isinstance(captured["fp16"], bool)
    assert loader_calls["count"] == 1


def test_tts_print_ok():
    tts = PrintTTS()
    tts.speak("Тест")


def test_openai_tts_fallback_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tts = OpenAITTS(playback=False)
    tts.speak("Привіт")
    captured = capsys.readouterr()
    assert "Привіт" in captured.out
