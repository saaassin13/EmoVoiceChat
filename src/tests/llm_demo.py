import sys
from pathlib import Path
# 添加项目根目录到 sys.path（用于直接运行此文件时）
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 使用示例
from src.llm import LLMEngine, LLMConfig
from src.context import ConversationHistory
from src.ser.schemas import EmotionLabel

# 1. 初始化对话历史
history_manager = ConversationHistory(user_id="test")

# 2. 配置并初始化 LLM
llm_config = LLMConfig()
llm_config.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
llm_config.load_in_4bit = True
llm_engine = LLMEngine(llm_config)

# 3. 生成回复
# user_text = "今天一点也不开心,工作也不顺利,代码也写的不好！"
user_text = "接下来你说话的语气都调整为生气，并且语气要严厉一点"
user_emotion = EmotionLabel.NEUTRAL
# user_emotion = EmotionLabel.HAPPY
history = history_manager.get_recent_turns()

reply = llm_engine.generate_reply(
    user_text=user_text,
    user_emotion=user_emotion,
    history=history
)

print(f"{reply}")

# # 4. 保存到历史
# history_manager.add_turn(
#     user_text=user_text,
#     user_emotion=user_emotion,
#     assistant_reply=reply
# )