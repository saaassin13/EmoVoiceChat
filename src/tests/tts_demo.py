import sys
from pathlib import Path
# 添加项目根目录到 sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tts.tts_engine import TTSEngine
from src.ser.schemas import EmotionLabel

# 初始化 TTS 引擎
tts = TTSEngine()

# 合成语音（根据情绪）
output_path = tts.synthesize(
    # text="今天的天气真好,我们去春游吧,我们带上帐篷和露营车,带上水果和零食,在草坪上晒太阳",
    text="这个事情我早就告诉你要这样处理，你为什么不听，非要让我生气！",
    emotion=EmotionLabel.ANGRY,
    speaker_wav="src/data/input_voice/happy.wav",
    speed=1.2
)

# 释放显存
tts.unload()


# https://viem-ccy.github.io/EMOVIE/dataset_release 参考语料

# from TTS.api import TTS
# import torch

# # 加载模型时自动下载（首次运行会下载，约4GB）
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# print("✅ XTTS-v2 模型下载并加载完成！")