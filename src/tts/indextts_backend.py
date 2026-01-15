"""IndexTTS HTTP 后端实现"""

from re import T
import sys
from pathlib import Path
from typing import Optional
import requests
import json

# 添加项目根目录到 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ser.schemas import EmotionLabel


class IndexTTSBackend:
    """IndexTTS HTTP 后端（通过 HTTP POST 调用 IndexTTS 服务）"""

    def __init__(
        self,
        service_url: str = "http://localhost:8000/tts",
        timeout: int = 300,
        default_temperature: float = 0.8,
        default_top_p: float = 0.8,
        verbose: bool = True,
    ) -> None:
        """
        初始化 IndexTTS HTTP 后端
        
        :param service_url: IndexTTS 服务地址
        :param timeout: 请求超时时间（秒）
        :param default_temperature: 默认 temperature 参数
        :param default_top_p: 默认 top_p 参数
        :param verbose: 是否输出详细信息
        """
        self.service_url = service_url
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.verbose = verbose
        
        print(f"IndexTTS HTTP 后端初始化")
        print(f"  服务地址: {self.service_url}")
        print(f"  超时时间: {self.timeout}秒")

    def synthesize(
        self,
        text: str,
        emotion: EmotionLabel,
        output_path: str,
        language: str = "zh",
        speaker_wav:  Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        通过 HTTP POST 调用 IndexTTS 服务合成语音
        
        :param text: 要合成的文本
        :param emotion: 情绪标签（可能通过 speaker_wav 体现）
        :param output_path: 输出音频文件路径
        :param language: 语言代码
        :param speaker_wav: 说话人音频提示文件路径
        :param speed: 语速（IndexTTS 可能不支持，保留接口兼容性）
        :param temperature: temperature 参数
        :param top_p: top_p 参数
        :param kwargs: 额外的参数
        :return: 输出文件路径
        """
        # 准备输出目录
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建请求数据
        request_data = {
            "text": text,
            "output_path": str(output_path),
            "verbose": self.verbose,
            # "temperature": temperature if temperature is not None else self.default_temperature,
            # "top_p": top_p if top_p is not None else self.default_top_p,
            "spk_audio_prompt": speaker_wav,
        }
        if emotion:
            emo_text = emotion.value
            if emo_text == "neutral":
                emo_text = "calm"
            request_data.update({
                "emo_text": emotion.value,
                "use_emo_text": True,
                "emo_alpha": 0.6
            })
        
        # 发送 POST 请求
        try:
            print(f"正在调用 IndexTTS 服务...")
            print(f"  文本: {text}")
            # print(f"  情绪: {emotion.value}")
            if speaker_wav:
                print(f"  说话人音频: {request_data.get('spk_audio_prompt')}")
            
            response = requests.post(
                self.service_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # IndexTTS 服务可能会返回 JSON 响应或直接保存文件
            # 根据实际服务实现调整
            if response.headers.get("Content-Type", "").startswith("application/json"):
                result = response.json()
                print(f"服务响应: {result}")
            else:
                print(f"请求成功，音频已保存到: {output_path}")
            
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"调用 IndexTTS 服务失败: {e}")

    def unload(self) -> None:
        """卸载后端（HTTP 后端无需卸载，保留接口兼容性）"""
        pass


if __name__ == "__main__":
    # 测试不同情绪
    # ['happy','angry', 'sad', 'calm']
    backend = IndexTTSBackend()
    for _emotion in [EmotionLabel.HAPPY, EmotionLabel.ANGRY]:
        test_text = f"这是一个测试语音，我当前很{_emotion.value}"

        
        output_file = f"/home/yanlan/workspace/code/emo-voice-chat/data/output_voice/test_{_emotion.value}.wav"
        speaker_wav = f"/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/happy.wav"
        result = backend.synthesize(test_text, _emotion, output_file, speaker_wav=speaker_wav)
        print(f"生成完成: {result}")