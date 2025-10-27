from __future__ import annotations

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class TTSEngineBase:
    """Base contract for TTS engines."""

    def speak(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None) -> None:
        raise NotImplementedError


class PrintTTS(TTSEngineBase):
    """Simple TTS that prints the phrase to stdout."""

    def speak(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None) -> None:
        label = voice or "TTS"
        print(f"[{label}] {text}")


class Pyttsx3TTS(TTSEngineBase):
    """Offline TTS using pyttsx3 with optional voice selection."""

    def __init__(self, voice: Optional[str] = None):
        self.voice = voice
        self._fallback = PrintTTS()
        try:
            import pyttsx3  # type: ignore

            self.engine = pyttsx3.init()
            if voice:
                with suppress(Exception):
                    self.engine.setProperty("voice", voice)
        except Exception as exc:  # pragma: no cover - pyttsx3 may not be available in tests
            log.warning("pyttsx3 unavailable (%s), falling back to console output", exc)
            self.engine = None

    def speak(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None) -> None:
        if self.engine is None:
            self._fallback.speak(text, voice=voice)
            return
        if voice and voice != self.voice:
            with suppress(Exception):
                self.engine.setProperty("voice", voice)
        self.engine.say(text)
        self.engine.runAndWait()


class OpenAITTS(TTSEngineBase):
    """Cloud TTS that uses the OpenAI speech synthesis API for realistic voices."""

    api_url = "https://api.openai.com/v1/audio/speech"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        playback: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.voice = voice or os.getenv("OPENAI_TTS_VOICE") or "alloy"
        self.model = model or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.language = language or os.getenv("OPENAI_TTS_LANGUAGE")
        self.playback = playback
        self._fallback = PrintTTS()
        try:  # pragma: no cover - executed in runtime environments with network access
            import requests  # type: ignore

            self._session = requests.Session()
        except Exception as exc:
            log.warning("Requests library unavailable (%s); TTS will fall back to console output", exc)
            self._session = None
        self._player = self._load_player()

    @staticmethod
    def _load_player():
        try:  # pragma: no cover - audio playback is best-effort
            import simpleaudio  # type: ignore

            return simpleaudio
        except Exception:
            return None

    @classmethod
    def is_configured(cls) -> bool:
        """Return True if credentials for the OpenAI voice API are available."""

        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import requests  # type: ignore # noqa: F401

            return True
        except Exception:
            return False

    def speak(self, text: str, *, voice: Optional[str] = None, language: Optional[str] = None) -> None:
        if not text:
            return
        if not self.api_key or self._session is None:
            self._fallback.speak(text, voice=voice)
            return
        payload = {
            "model": self.model,
            "voice": voice or self.voice,
            "input": text,
            "format": "wav",
        }
        lang = language or self.language
        if lang:
            payload["language"] = lang
        try:  # pragma: no cover - network calls not executed in tests
            response = self._session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            audio_bytes = response.content
        except Exception as exc:
            log.warning("OpenAI TTS unavailable (%s); falling back to console output", exc)
            self._fallback.speak(text, voice=voice)
            return
        if not audio_bytes:
            self._fallback.speak(text, voice=voice)
            return
        self._play_audio(audio_bytes, voice or self.voice)

    def _play_audio(self, audio_bytes: bytes, voice: str) -> None:
        if not audio_bytes:
            return
        tmp_path = self._persist_audio(audio_bytes)
        if self.playback and self._player is not None:
            try:  # pragma: no cover - depends on host audio stack
                wave = self._player.WaveObject.from_wave_file(str(tmp_path))
                play_obj = wave.play()
                play_obj.wait_done()
                tmp_path.unlink(missing_ok=True)
                return
            except Exception as exc:
                log.warning("Audio playback failed (%s); keeping synthesized file", exc)
        print(f"[{voice}] Аудіо збережено у {tmp_path}")

    @staticmethod
    def _persist_audio(audio_bytes: bytes) -> Path:
        fd, path = os.mkstemp(prefix="vct-openai-", suffix=".wav")
        with os.fdopen(fd, "wb") as fh:
            fh.write(audio_bytes)
        return Path(path)
