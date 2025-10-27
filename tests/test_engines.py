from vct.engines.stt import RuleBasedSTT
from vct.engines.tts import PrintTTS

def test_rulebased_stt_mapping():
    stt = RuleBasedSTT()
    assert stt.transcribe(wav_path="data/examples/commands/sydity.wav") == "сидіти"

def test_tts_print_ok():
    tts = PrintTTS()
    tts.speak("Тест")