"""可选 VITS 后端实现"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional
import torch
import numpy as np
import soundfile as sf

from src.tts.config import (
    VITS_MODEL_NAME,
    DEFAULT_DEVICE,
    SAMPLE_RATE,
    EmotionTTSParams,
    get_emotion_params,
)
from src.ser.schemas import EmotionLabel


class VITSBackend:
    """VITS 后端（轻量版，支持情绪音色切换）"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 VITS 后端
        
        :param model_name: 模型名称或路径，None 则使用默认配置
        :param device: 设备（cuda/cpu），None 则自动选择
        """
        self.model_name = model_name or VITS_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # VITS 模型加载（这里使用 Coqui TTS 的 VITS 模型）
        # 如果需要使用独立的 VITS 实现，可以替换为其他库
        print(f"正在加载 VITS 模型: {self.model_name}")
        print(f"  设备: {self.device}")
        
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name, progress_bar=True)
            if self.device == "cuda" and torch.cuda.is_available():
                self.tts.to(self.device)
            print(f"模型加载成功")
        except Exception as e:
            raise RuntimeError(f"无法加载 VITS 模型: {e}")

    def synthesize(
        self,
        text: str,
        emotion: EmotionLabel,
        output_path: str,
        language="zh",
        speaker_wav=None,
        speed=1.0,
        **kwargs
    ) -> str:
        """
        合成语音（根据情绪切换音色）
        
        :param text: 要合成的文本
        :param emotion: 情绪标签
        :param output_path: 输出音频文件路径
        :param kwargs: 额外的 TTS 参数
        :return: 输出文件路径
        """
        # 获取情绪对应的 TTS 参数
        params = get_emotion_params(emotion).to_dict()
        
        # 准备输出目录
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # VITS 模型通常支持 speaker_id 或 emotion 参数
        # 根据实际模型调整
        tts_kwargs = {
            "text": text,
            "file_path": str(output_path),
            "language": language,
            "speaker_wav": speaker_wav,
            "speed": speed
        }
        
        # 如果模型支持 speaker_id，可以根据情绪选择不同的说话人
        # 例如：happy -> speaker_id=0, sad -> speaker_id=1 等
        emotion_speaker_map = {
            EmotionLabel.HAPPY: 0,
            EmotionLabel.SAD: 1,
            EmotionLabel.ANGRY: 2,
            EmotionLabel.NEUTRAL: 3,
        }
        
        # 尝试添加 speaker_id（如果模型支持）
        if hasattr(self.tts, 'speaker_ids') and self.tts.speaker_ids:
            speaker_id = emotion_speaker_map.get(emotion, 3)
            tts_kwargs["speaker"] = self.tts.speaker_ids[speaker_id % len(self.tts.speaker_ids)]
        
        # 合成语音
        self.tts.tts_to_file(**tts_kwargs)
        
        return str(output_path)

    def unload(self) -> None:
        """卸载模型（释放显存）"""
        if hasattr(self, 'tts'):
            del self.tts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # 测试
    backend = VITSBackend()
    
    test_text = "你好，这是一个测试。"
    output_path = "data/output_voice/test_vits.wav"
    
    # 测试不同情绪
    for emotion in [EmotionLabel.NEUTRAL, EmotionLabel.HAPPY]:
        output_file = output_path.replace(".wav", f"_{emotion.value}.wav")
        result = backend.synthesize(test_text, emotion, output_file)
        print(f"生成完成: {result} (情绪: {emotion.value})")
    
    backend.unload()