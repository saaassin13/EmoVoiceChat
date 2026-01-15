"""Coqui TTS emotion-tts 后端实现"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional
import torch
from TTS.api import TTS

from src.tts.config import (
    COQUI_MODEL_NAME,
    DEFAULT_DEVICE,
    SAMPLE_RATE,
    EmotionTTSParams,
    get_emotion_params,
)
from src.ser.schemas import EmotionLabel


class CoquiTTSBackend:
    """Coqui TTS 后端（支持情绪语音生成）"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 Coqui TTS 后端
        
        :param model_name: 模型名称或路径，None 则使用默认配置
        :param device: 设备（cuda/cpu），None 则自动选择
        """
        self.model_name = model_name or COQUI_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 TTS 模型
        print(f"正在加载 Coqui TTS 模型: {self.model_name}")
        print(f"  设备: {self.device}")
        
        try:
            self.tts = TTS(model_name=self.model_name, progress_bar=True)
            # 如果支持 CUDA，移动到 GPU
            if self.device == "cuda" and torch.cuda.is_available():
                self.tts.to(self.device)
            print(f"模型加载成功")
        except Exception as e:
            print(f"警告：加载模型失败: {e}")
            # print("尝试使用默认模型...")
            # 尝试使用更通用的模型
            # try:
            #     self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
            #     if self.device == "cuda" and torch.cuda.is_available():
            #         self.tts.to(self.device)
            # except Exception as e2:
            #     raise RuntimeError(f"无法加载 TTS 模型: {e2}")

            raise e

    def synthesize(
        self,
        text: str,
        emotion: EmotionLabel,
        output_path: str,
        **kwargs
    ) -> str:
        """
        合成语音（根据情绪调整参数）
        
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
        
        # 构建 TTS 参数
        tts_kwargs = {
            "text": text,
            "file_path": str(output_path),
            "kwargs": params,
            
        }
        
        # 根据模型支持的功能添加参数
        # 注意：不同模型支持的参数可能不同，需要根据实际模型调整
    
        self.tts.tts_to_file(**tts_kwargs)
        
        # 后处理：应用语速、音高调整（如果需要）
        # 注意：Coqui TTS 可能不支持直接调整这些参数
        # 如果需要，可以使用 librosa 或 pydub 进行后处理
        
        return str(output_path)

    def unload(self) -> None:
        """卸载模型（释放显存）"""
        if hasattr(self, 'tts'):
            del self.tts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # 测试
    backend = CoquiTTSBackend()
    
    test_text = "你好，这是一个测试。"
    output_path = "data/output_voice/test_coqui.wav"
    
    # 测试不同情绪
    for emotion in [EmotionLabel.NEUTRAL, EmotionLabel.HAPPY, EmotionLabel.SAD, EmotionLabel.ANGRY]:
        output_file = output_path.replace(".wav", f"_{emotion.value}.wav")
        result = backend.synthesize(test_text, emotion, output_file)
        print(f"生成完成: {result} (情绪: {emotion.value})")
    
    backend.unload()