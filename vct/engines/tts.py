class TTSEngineBase:
    def speak(self, text: str) -> None: raise NotImplementedError

class Pyttsx3TTS(TTSEngineBase):
    def __init__(self):
        try:
            import pyttsx3  # type: ignore
            self.engine = pyttsx3.init()
        except Exception:
            self.engine = None
    def speak(self, text: str) -> None:
        if self.engine is None: print(f"[TTS] {text}")
        else:
            self.engine.say(text); self.engine.runAndWait()

class PrintTTS(TTSEngineBase):
    def speak(self, text: str) -> None: print(f"[TTS] {text}")