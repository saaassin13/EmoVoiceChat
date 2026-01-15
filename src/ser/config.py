"""置信度阈值、模型路径等"""

import os
from typing import Literal

# 模型选择：使用基于 wav2vec2 的情绪识别模型
# 如果 Hugging Face 上有专门的 wav2vec2-base-emo，替换这里的模型名
# 这里使用一个通用的情绪识别模型作为示例
SER_MODEL_NAME: str = os.getenv(
    "SER_MODEL_NAME",
    "facebook/wav2vec2-base"  # 可以替换为实际的情绪识别模型
)

# 置信度阈值：低于此值则标记为 neutral
CONFIDENCE_THRESHOLD: float = float(os.getenv("SER_CONFIDENCE_THRESHOLD", "0.7"))

# 设备选择
SERDevice = Literal["cuda", "cpu"]
DEFAULT_DEVICE: SERDevice = "cuda" if os.getenv("SER_DEVICE", "") in ["cuda", "cpu"] else "cuda"

# 音频采样率（与 Whisper 保持一致）
SAMPLE_RATE: int = 16_000