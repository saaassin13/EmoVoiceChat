"""
封装 SenseVoiceSmall 调用（加载模型、转写、同时返回文本和情绪）
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import os
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


# 默认模型目录，可以按需改成你的路径或用环境变量覆盖
DEFAULT_SENSEVOICE_MODEL_DIR: str = os.getenv(
    "SENSEVOICE_MODEL_DIR",
    "/home/yanlan/.cache/modelscope/hub/models/iic/SenseVoiceSmall",
)

# 默认设备：优先 cuda:0，没有就用 cpu
DEFAULT_SENSEVOICE_DEVICE: str = os.getenv("SENSEVOICE_DEVICE", "cuda:0")


@dataclass
class SenseVoiceResult:
    text: str
    emotion: Optional[str]
    language: Optional[str]
    raw: Dict[str, Any]


class SenseVoiceEngine:
    """SenseVoice 语音转文字 + 情绪识别封装"""

    def __init__(
        self,
        model_dir: str = DEFAULT_SENSEVOICE_MODEL_DIR,
        device: str = DEFAULT_SENSEVOICE_DEVICE,
    ) -> None:
        self.model_dir = model_dir
        self.device = device

        self._model = AutoModel(
            model=self.model_dir,
            trust_remote_code=False,
            remote_code="./model.py",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
        )

    @staticmethod
    def _parse_text_with_tags(text_raw: str) -> tuple[str, Optional[str], Optional[str]]:
        """
        解析包含标签的文本字符串，提取语言、情绪和纯文本。
        
        格式示例: '<|zh|><|NEUTRAL|><|Speech|><|withitn|>实际文本内容'
        
        :param text_raw: 原始文本字符串
        :return: (纯文本, 语言, 情绪)
        """
        if not text_raw:
            return "", None, None
        
        # 提取语言标签: <|zh|>, <|en|> 等
        lang_match = re.search(r'<\|([a-z]+)\|>', text_raw)
        language = lang_match.group(1) if lang_match else None
        
        # 提取情绪标签: <|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|> 等
        emotion_match = re.search(r'<\|(HAPPY|SAD|ANGRY|NEUTRAL)\|>', text_raw)
        emotion = emotion_match.group(1) if emotion_match else None
        
        # 移除所有 <|...|> 标签，提取纯文本
        text_clean = re.sub(r'<\|[^|]+\|>', '', text_raw).strip()
        
        return text_clean, language, emotion

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        use_itn: bool = True,
        batch_size_s: int = 60,
        merge_vad: bool = True,
        merge_length_s: int = 15,
    ) -> SenseVoiceResult:
        """
        对音频进行转写，并抽取文本和情绪。

        :param audio_path: 音频文件路径
        """
        res: List[Dict[str, Any]] = self._model.generate(
            input=audio_path,
            cache={},
            language=language,      # "zn", "en", "yue", "ja", "ko", "nospeech", "auto"
            use_itn=use_itn,
            batch_size_s=batch_size_s,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
        )

        if not res:
            return SenseVoiceResult(
                text="",
                emotion=None,
                language=None,
                raw={"results": res},
            )

        first = res[0]

        # 获取原始文本字符串
        text_raw = first.get("text", "") or ""
        
        # 解析文本中的标签，提取语言、情绪和纯文本
        text_clean, language_out, emotion = self._parse_text_with_tags(text_raw)
        
        # 对纯文本进行后处理（ITN等）
        text = rich_transcription_postprocess(text_clean)

        return SenseVoiceResult(
            text=text,
            emotion=emotion,
            language=language_out,
            raw=first,
        )


if __name__ == "__main__":
    audio = "/home/yanlan/workspace/ai/index-tts/data/voice_clone_calm.wav"
    engine = SenseVoiceEngine()
    result = engine.transcribe(audio)

    print("\n========== SenseVoice 结果 ==========")
    print("文本:", result.text)
    print("情绪:", result.emotion)
    print("语言:", result.language)
    print("====================================")