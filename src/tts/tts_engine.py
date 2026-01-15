"""统一 synthesize(text, emotion) -> wav_path 接口"""

import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tts.config import (
    DEFAULT_BACKEND,
    TTSBackendType,
    OUTPUT_DIR,
    OUTPUT_FORMAT,
)
from src.tts.coqui_backend import CoquiTTSBackend
from src.tts.vits_backend import VITSBackend
from src.tts.indextts_backend import IndexTTSBackend
from src.ser.schemas import EmotionLabel


class TTSEngine:
    """统一的 TTS 引擎接口"""

    def __init__(
        self,
        backend_type: Optional[TTSBackendType] = None,
        **backend_kwargs
    ) -> None:
        """
        初始化 TTS 引擎
        
        :param backend_type: 后端类型（coqui/vits），None 则使用默认配置
        :param backend_kwargs: 传递给后端的额外参数
        """
        self.backend_type = backend_type or DEFAULT_BACKEND
        self.backend = self._create_backend(**backend_kwargs)
        
        # 确保输出目录存在
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _create_backend(self, **kwargs):
        """根据配置创建对应的后端"""
        if self.backend_type == "coqui":
            return CoquiTTSBackend(**kwargs)
        elif self.backend_type == "vits":
            return VITSBackend(**kwargs)
        elif self.backend_type == "indextts":
            return IndexTTSBackend(**kwargs)
        else:
            raise ValueError(f"不支持的后端类型: {self.backend_type}")

    def synthesize(
        self,
        text: str,
        emotion: EmotionLabel,
        output_path: Optional[str] = None,
        language = "zh",
        speaker_wav = None,
        speed = 1.0,
        **kwargs
    ) -> str:
        """
        合成语音（统一接口）
        
        :param text: 要合成的文本
        :param emotion: 情绪标签
        :param output_path: 输出音频文件路径，None 则自动生成
        :param kwargs: 额外的 TTS 参数
        :return: 输出文件路径
        """
        # 如果没有指定输出路径，自动生成
        if output_path is None:
            import time
            timestamp = int(time.time())
            filename = f"tts_{emotion.value}_{timestamp}.{OUTPUT_FORMAT}"
            output_path = str(Path(OUTPUT_DIR) / filename)
        
        # 调用后端合成
        result_path = self.backend.synthesize(
            text=text,
            emotion=emotion,
            output_path=output_path,
            language=language,
            speaker_wav=speaker_wav,
            speed=speed,
            **kwargs
        )
        
        return result_path

    def unload(self) -> None:
        """卸载模型（释放显存）"""
        self.backend.unload()


if __name__ == "__main__":
    # 测试
    # # {'happy': 0.0, 'angry': 0.85, 'sad': 0.02, 'afraid': 0.05, 'disgusted': 0.05, 'melancholic': 0.02, 'surprised': 0.0, 'calm': 0.01}
    tts = TTSEngine()
    _speaker_wav = f"/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/happy.wav"
    
    test_cases = [
        ("你好，很高兴见到你！", EmotionLabel.HAPPY),
        ("我很难过，心情不好。", EmotionLabel.SAD),
        ("这太让人生气了！", EmotionLabel.ANGRY),
        # ("今天天气不错。", EmotionLabel.NEUTRAL),
    ]
    
    print("\n========== TTS 测试 ==========")
    for text, emotion in test_cases:
        # (test_text, _emotion, output_file, speaker_wav=speaker_wav)
        output_path = tts.synthesize(text, emotion, speaker_wav=_speaker_wav)
        print(f"文本: {text}")
        print(f"情绪: {emotion.value}")
        print(f"输出: {output_path}")
        print("-" * 40)
    
    tts.unload()
    print("测试完成！")