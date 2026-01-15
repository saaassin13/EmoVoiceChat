"""TTS 模型配置：情绪 → TTS 参数（语速、音高、风格）映射"""

import os
from typing import Literal, Dict
from dataclasses import dataclass

from src.config.settings import ROOT_DIR
from src.ser.schemas import EmotionLabel

# TTS 后端类型
TTSBackendType = Literal["coqui", "vits", "indextts"]

# 默认后端（Coqui TTS）
DEFAULT_BACKEND: TTSBackendType = os.getenv("TTS_BACKEND", "indextts")

# IndexTTS HTTP 服务配置
# model:https://modelscope.cn/models/IndexTeam/IndexTTS-2
INDEXTTS_SERVICE_URL: str = os.getenv(
    "INDEXTTS_SERVICE_URL",
    "http://localhost:8000/tts"
)
INDEXTTS_TIMEOUT: int = int(os.getenv("INDEXTTS_TIMEOUT", "300"))
INDEXTTS_TEMPERATURE: float = float(os.getenv("INDEXTTS_TEMPERATURE", "0.8"))
INDEXTTS_TOP_P: float = float(os.getenv("INDEXTTS_TOP_P", "0.8"))

# Coqui TTS 配置
# tts_models/zh-CN/baker/tacotron2-DDC-GST
# tts_models/zh-CN/emotion-tts/vits
# tts_models/multilingual/multi-dataset/xtts_v2
COQUI_MODEL_NAME: str = os.getenv(
    "COQUI_MODEL_NAME",
    "tts_models/zh-CN/emotion-tts/vits"  # 中文模型
)

# VITS 配置（可选）
VITS_MODEL_NAME: str = os.getenv(
    "VITS_MODEL_NAME",
    "tts_models/multilingual/multi-dataset/xtts_v2"  # 默认使用 Coqui 模型
)

# 设备选择
TTSDevice = Literal["cuda", "cpu"]
DEFAULT_DEVICE: TTSDevice = "cuda" if os.getenv("TTS_DEVICE", "") in ["cuda", "cpu"] else "cuda"

# 音频采样率（与 Whisper/SER 保持一致）
SAMPLE_RATE: int = 16_000

# 输出音频格式
OUTPUT_FORMAT: str = os.getenv("TTS_OUTPUT_FORMAT", "wav")

# 输出目录
OUTPUT_DIR: str = os.getenv("TTS_OUTPUT_DIR", os.path.join(ROOT_DIR, "data/output_voice"))


@dataclass
class EmotionTTSParams:
    """情绪对应的 TTS 参数"""
    speed: float  # 语速倍数（1.0 为正常，>1.0 更快，<1.0 更慢）
    pitch: float  # 音高调整（0.0 为正常，>0.0 更高，<0.0 更低）
    volume: float  # 音量（0.0-1.0）
    emotion_style: str  # 情绪风格标签（用于 Coqui emotion-tts）

    def to_dict(self):
        return {
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "emotion": self.emotion_style,
        }


# 情绪 → TTS 参数映射表（根据 plan.md 第 110-117 行）
EMOTION_TTS_MAPPING: Dict[EmotionLabel, EmotionTTSParams] = {
    EmotionLabel.ANGRY: EmotionTTSParams(
        speed=0.85,  # 降低语速
        pitch=-0.1,  # 调低音调
        volume=0.9,
        emotion_style="angry"  # 温和、耐心的语调
    ),
    EmotionLabel.HAPPY: EmotionTTSParams(
        speed=1.15,  # 加快语速
        pitch=0.15,  # 调高音调
        volume=1.0,
        emotion_style="happy"  # 活泼、热情的语调
    ),
    EmotionLabel.SAD: EmotionTTSParams(
        speed=0.9,   # 放缓语速
        pitch=-0.2,  # 低沉音调
        volume=0.85,
        emotion_style="sad"  # 温柔、安慰的语调
    ),
    EmotionLabel.NEUTRAL: EmotionTTSParams(
        speed=1.0,   # 正常语速
        pitch=0.0,  # 平稳音调
        volume=0.95,
        emotion_style="neutral"  # 自然、流畅的语调
    ),
}


def get_emotion_params(emotion: EmotionLabel) -> EmotionTTSParams:
    """获取指定情绪对应的 TTS 参数"""
    return EMOTION_TTS_MAPPING.get(emotion, EMOTION_TTS_MAPPING[EmotionLabel.NEUTRAL])