"""单轮对话数据结构：轮次 id、用户文本、情绪、助手回复、时间戳"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.ser.schemas import EmotionLabel


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    turn_id: int  # 轮次ID（从1开始）
    user_id: str  # 用户ID（新增）
    user_text: str  # 用户输入的文本
    user_emotion: EmotionLabel  # 用户情绪标签
    assistant_reply: str  # 助手回复的文本
    timestamp: datetime  # 时间戳
    
    def to_dict(self) -> dict:
        """转换为字典（用于JSON序列化）"""
        return {
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "user_text": self.user_text,
            "user_emotion": self.user_emotion.value,
            "assistant_reply": self.assistant_reply,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """从字典创建对象（用于JSON反序列化）"""
        return cls(
            turn_id=data["turn_id"],
            user_id=data.get("user_id", "anonymous"),  # 兼容旧数据（虽然不需要，但更安全）
            user_text=data["user_text"],
            user_emotion=EmotionLabel(data["user_emotion"]),
            assistant_reply=data["assistant_reply"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )