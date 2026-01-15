"""情绪标签枚举、置信度结构"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class EmotionLabel(str, Enum):
    """情绪标签枚举（4 类基础情绪）"""
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    NEUTRAL = "neutral"


@dataclass
class SERResult:
    """SER 识别结果"""
    emotion: EmotionLabel  # 最终情绪标签
    confidence: float  # 置信度（0-1）
    raw_scores: Dict[str, float]  # 所有情绪的原始得分
    raw: Dict[str, Any]  # 原始模型输出（用于调试）