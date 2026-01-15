"""封装 whisper 调用（加载模型、转写、语言固定为中文）"""
import sys
from pathlib import Path

# 添加项目根目录到 sys.path（用于直接运行此文件时）
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import whisper

from src.stt.config import (
    WHISPER_MODEL_NAME,
    WHISPER_LANGUAGE,
    DEFAULT_DEVICE,
    USE_FP16,
    BEAM_SIZE,
)


@dataclass
class STTResult:
    text: str
    language: str
    raw: Dict[str, Any]


class WhisperEngine:
    """Whisper 语音转文字封装"""

    def __init__(
        self,
        model_name: str = WHISPER_MODEL_NAME,
        device: Optional[str] = None,
        language: str = WHISPER_LANGUAGE,
        use_fp16: bool = USE_FP16,
        beam_size: int = BEAM_SIZE,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device or ( "cuda" if torch.cuda.is_available() else "cpu" )
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.beam_size = beam_size

        # 初次下载后续缓存到：~/.cache/whisper 
        self._model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
    ) -> STTResult:
        """
        对给定 wav/mp3/flac 等音频文件进行转写。

        :param audio_path: 音频文件路径（推荐 16kHz 单通道 wav）
        :param prompt: 可选的引导提示（如场景、专有名词）
        """
        options = dict(
            language=self.language,
            beam_size=self.beam_size,
            fp16=self.use_fp16,
        )
        if prompt:
            options["initial_prompt"] = prompt

        result = self._model.transcribe(audio_path, **options)

        # result 里包含 "text"、"segments"、"language" 等
        return STTResult(
            text=result.get("text", "").strip(),
            language=result.get("language", self.language),
            raw=result,
        )


if __name__ == "__main__":
    out_path = "/home/yanlan/workspace/code/emo-voice-chat/data/output_voice/test_angry.wav"
    # out_path = "src/data/input_voice/demo.wav"
    stt = WhisperEngine()
    result = stt.transcribe(out_path)

    print("\n========== 转写结果 ==========")
    print(result.text)
    print("================================")