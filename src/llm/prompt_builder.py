"""根据文档里的模板构造 prompt（含情绪策略）"""

from typing import List

from src.context.schemas import ConversationTurn
from src.ser.schemas import EmotionLabel


class PromptBuilder:
    """根据情绪和对话历史构造 LLM Prompt"""
    
    # 情绪策略说明
    EMOTION_STRATEGIES = {
        EmotionLabel.ANGRY: "若用户情绪为angry（愤怒）：回复需温和、安抚，避免激化矛盾；",
        EmotionLabel.HAPPY: "若用户情绪为happy（开心）：回复需热情、轻快，呼应用户情绪；",
        EmotionLabel.SAD: "若用户情绪为sad（悲伤）：回复需共情、鼓励；",
        EmotionLabel.NEUTRAL: "若用户情绪为neutral（中性）：回复需自然、简洁。",
    }
    
    @staticmethod
    def build_prompt(
        user_text: str,
        user_emotion: EmotionLabel,
        history: List[ConversationTurn]
    ) -> str:
        """
        构造完整的 Prompt
        
        :param user_text: 当前用户输入的文本
        :param user_emotion: 当前用户情绪标签
        :param history: 对话历史列表
        :return: 构造好的 Prompt 字符串
        """
        # 1. 系统提示词
        system_prompt = """你是一个贴心的语音聊天助手，需要根据用户的情绪组织合适的回复并且使用合适的语气：

1. 若用户情绪为angry（愤怒）：回复需温和、安抚，避免激化矛盾；
2. 若用户情绪为happy（开心）：回复需热情、轻快，呼应用户情绪；
3. 若用户情绪为sad（悲伤）：回复需共情、鼓励；
4. 若用户情绪为neutral（中性）：回复需自然、简洁。
"""
        
        # 2. 对话历史
        history_text = ""
        if history:
            history_text = "对话历史：\n\n"
            for turn in history:
                history_text += f"用户（{turn.user_emotion.value}）：{turn.user_text}\n"
                history_text += f"助手：{turn.assistant_reply}\n\n"
        
        # 3. 当前输入
        current_input = f"""当前用户情绪：{user_emotion.value}

当前用户输入：{user_text}
"""

        result_input = """请结合上述信息，生成一个最合适的回复。

        现在开始，请严格遵守以下要求：
        1. 只输出一个 JSON 对象，不要输出任何解释、说明、前后缀或多余文字。
        2. 不要输出 ```、不需要输出“JSON:”等提示。
        3. 如果无法严格满足要求，也仍然要输出一个合法的 JSON，并在 replay 字段中说明情况。

        输出格式（请照抄字段名）：
        {"replay": "你的回复内容", "emotion": "calm"}"""
        
        # 4. 组合完整 Prompt
        full_prompt = system_prompt + history_text + current_input + result_input
        
        return full_prompt
    
    @staticmethod
    def build_simple_prompt(
        user_text: str,
        user_emotion: EmotionLabel,
        history: List[ConversationTurn]
    ) -> str:
        """
        构造简化版 Prompt（适用于某些模型格式要求）
        
        :param user_text: 当前用户输入的文本
        :param user_emotion: 当前用户情绪标签
        :param history: 对话历史列表
        :return: 构造好的 Prompt 字符串
        """
        # 构建历史对话文本
        history_parts = []
        for turn in history:
            history_parts.append(f"用户（{turn.user_emotion.value}）：{turn.user_text}")
            history_parts.append(f"助手：{turn.assistant_reply}")
        
        # 当前输入
        current_part = f"用户（{user_emotion.value}）：{user_text}"
        
        # 组合
        if history_parts:
            context = "\n".join(history_parts) + "\n" + current_part
        else:
            context = current_part
        
        # 添加情绪指导
        emotion_guide = PromptBuilder.EMOTION_STRATEGIES.get(user_emotion, "")
        
        prompt = f"""你是一个贴心的语音聊天助手。{emotion_guide}

请根据以下对话生成回复：

{context}

助手："""
        
        return prompt

    @staticmethod
    def build_chat_messages(
        user_text: str,
        user_emotion: EmotionLabel,
        history: List[ConversationTurn]
    ) -> list:
        """构造适用于 Qwen chat_template 的 messages 列表"""

        # 1) 系统提示词：含情绪策略
        system_content = """你是一个贴心的语音聊天助手，需要根据用户的情绪组织合适的回复并且使用合适的语气：
1. 若用户情绪为angry（愤怒）：回复需温和、安抚，避免激化矛盾；
2. 若用户情绪为happy（开心）：回复需热情、轻快，呼应用户情绪；
3. 若用户情绪为sad（悲伤）：回复需共情、鼓励；
4. 若用户情绪为neutral（中性）：回复需自然、简洁。"""

        # 2) 构造历史对话文本
        history_text = ""
        if history:
            history_text = "对话历史：\n\n"
            for turn in history:
                history_text += f"用户（{turn.user_emotion.value}）：{turn.user_text}\n"
                history_text += f"助手：{turn.assistant_reply}\n\n"

        # 3) 当前轮输入 + JSON 格式约束
        user_part = f"""当前用户情绪：{user_emotion.value}

当前用户输入：{user_text}

请结合上述信息，生成一个最合适的回复。

严格遵守以下规则：
1. 你只能输出一个 JSON 对象；
2. 不要输出任何多余的文字、说明、前缀或后缀；
3. 不要输出 ``` 符号，不要输出“JSON:”等提示。
4. 情绪标签为其中一种:['happy', 'angry', 'calm', 'surprised']

输出格式（字段名必须完全一致）：
{{"replay": "你的回复内容", "emotion": "情绪标签"}}"""

        # 最终 user content：历史 + 当前轮
        full_user_content = history_text + user_part

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_user_content},
        ]
        return messages