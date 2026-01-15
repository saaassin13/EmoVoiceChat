import sys
from pathlib import Path
# 添加项目根目录到 sys.path（用于直接运行此文件时）
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.stt.sensevoice_engine import SenseVoiceEngine

audio_path = "/home/yanlan/workspace/ai/index-tts/data/voice_clone_calm.wav"

engine = SenseVoiceEngine()
result = engine.transcribe(audio_path)

print("文本:", result.text)
print("情绪:", result.emotion)
print("语言:", result.language)