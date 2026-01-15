"""Whisper 模型配置：模型名称、设备、语言等"""

from typing import Literal
import os

# 默认使用的模型（推荐 large-v3，3090 够用）
# 你也可以根据显存改成： "medium" / "small" / "base"
WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "large-v3")

# 语言：固定中文，减少多语言干扰，提升准确率
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "zh")

# 设备选择：优先 CUDA，没有就用 CPU
WhisperDevice = Literal["cuda", "cpu"]

DEFAULT_DEVICE: WhisperDevice = "cuda" if os.getenv("WHISPER_DEVICE", "") in ["cuda", "cpu"] else "cuda"

# 是否使用 fp16（GPU 上推荐开启）
USE_FP16: bool = os.getenv("WHISPER_FP16", "1") == "1"

# beam search 宽度（越大越准但更慢）
BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

# 每次处理的最大音频长度（秒），长音频可用于后续分段
MAX_AUDIO_SECONDS: int = int(os.getenv("WHISPER_MAX_AUDIO_SECONDS", "120"))