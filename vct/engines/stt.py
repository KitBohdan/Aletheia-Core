"""Speech-to-text engines used by RoboDog."""

from __future__ import annotations

from typing import Callable, Optional


class STTEngineBase:
    """Abstract interface for speech-to-text backends."""

    def transcribe(self, wav_path: Optional[str] = None, use_mic: bool = False) -> str:
        """Return text transcription for the given audio clip."""

        raise NotImplementedError


class RuleBasedSTT(STTEngineBase):
    """Legacy keyword-based recogniser kept as a lightweight fallback."""

    KEYWORDS = {"sydity": "сидіти", "lezhaty": "лежати", "do_mene": "до мене", "bark": "голос"}

    def transcribe(self, wav_path: Optional[str] = None, use_mic: bool = False) -> str:
        if use_mic or not wav_path:
            return ""
        name = str(wav_path).lower()
        for k, v in self.KEYWORDS.items():
            if k in name:
                return v
        return ""


class WhisperSTT(STTEngineBase):
    """Wrapper around OpenAI Whisper for high quality speech recognition."""

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        loader: Optional[Callable[[], "_WhisperModel"]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._loader = loader
        self._model: Optional[_WhisperModel] = None

    def _load_model(self) -> "_WhisperModel":
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in runtime, hard to test
            raise RuntimeError(
                "The openai-whisper package is required for WhisperSTT. "
                "Install it with `pip install openai-whisper`."
            ) from exc
        return whisper.load_model(self.model_name, device=self.device)

    def _ensure_model(self) -> "_WhisperModel":
        if self._model is None:
            loader = self._loader or self._load_model
            self._model = loader()
        return self._model

    @staticmethod
    def _supports_fp16() -> bool:
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - dependency may be absent during tests
            return False
        return torch.cuda.is_available()

    def transcribe(self, wav_path: Optional[str] = None, use_mic: bool = False) -> str:
        if use_mic:
            raise NotImplementedError("Microphone capture is not implemented for WhisperSTT")
        if not wav_path:
            return ""
        model = self._ensure_model()
        result = model.transcribe(str(wav_path), fp16=self._supports_fp16())
        text = result.get("text", "") if isinstance(result, dict) else ""
        return text.strip()


class _WhisperModel:
    """Protocol-like duck type for whisper model objects."""

    def transcribe(self, wav_path: str, *, fp16: bool) -> dict:
        raise NotImplementedError
