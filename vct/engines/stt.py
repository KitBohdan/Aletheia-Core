from typing import Optional

class STTEngineBase:
    def transcribe(self, wav_path: Optional[str] = None, use_mic: bool = False) -> str:
        raise NotImplementedError

class RuleBasedSTT(STTEngineBase):
    KEYWORDS = {"sydity": "сидіти", "lezhaty": "лежати", "do_mene": "до мене", "bark": "голос"}
    def transcribe(self, wav_path: Optional[str] = None, use_mic: bool = False) -> str:
        if use_mic or not wav_path:
            return ""
        name = str(wav_path).lower()
        for k, v in self.KEYWORDS.items():
            if k in name:
                return v
        return ""