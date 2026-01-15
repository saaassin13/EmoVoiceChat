"""读写本地 JSON，对话轮次的增删改查"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.context.config import MAX_HISTORY_TURNS, HISTORY_DIR
from src.context.schemas import ConversationTurn
from src.ser.schemas import EmotionLabel


class ConversationHistory:
    """对话历史管理器（支持多用户）"""
    
    def __init__(self, user_id: Optional[str] = None, history_dir: Optional[Path] = None):
        """
        初始化对话历史管理器
        
        :param user_id: 用户ID，如果为None则使用 "anonymous"
        :param history_dir: 历史文件目录，默认使用配置中的目录
        """
        self.user_id = user_id or "anonymous"
        self.history_dir = history_dir or HISTORY_DIR
        self.history_file = self.history_dir / f"{self.user_id}.json"
        
        # 确保目录存在
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self._turns: List[ConversationTurn] = []
        self._load()
    
    def _load(self) -> None:
        """从文件加载对话历史"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._turns = [
                        ConversationTurn.from_dict(turn_data)
                        for turn_data in data.get("turns", [])
                    ]
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"警告：加载用户 {self.user_id} 的对话历史失败，将使用空历史。错误：{e}")
                self._turns = []
        else:
            self._turns = []
    
    def _save(self) -> None:
        """保存对话历史到文件"""
        data = {
            "user_id": self.user_id,
            "turns": [turn.to_dict() for turn in self._turns]
        }
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_turn(
        self,
        user_text: str,
        user_emotion: EmotionLabel,
        assistant_reply: str
    ) -> ConversationTurn:
        """
        添加一轮对话
        
        :param user_text: 用户输入的文本
        :param user_emotion: 用户情绪标签
        :param assistant_reply: 助手回复的文本
        :return: 创建的对话轮次对象
        """
        turn_id = len(self._turns) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            user_id=self.user_id,
            user_text=user_text,
            user_emotion=user_emotion,
            assistant_reply=assistant_reply,
            timestamp=datetime.now()
        )
        self._turns.append(turn)
        
        # 限制历史轮数
        if len(self._turns) > MAX_HISTORY_TURNS:
            self._turns = self._turns[-MAX_HISTORY_TURNS:]
            # 重新编号
            for i, t in enumerate(self._turns, start=1):
                t.turn_id = i
        
        self._save()
        return turn
    
    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        获取最近的对话轮次
        
        :param n: 返回的轮次数，None表示返回全部
        :return: 对话轮次列表
        """
        if n is None:
            return self._turns.copy()
        return self._turns[-n:] if n > 0 else []
    
    def clear(self) -> None:
        """清空对话历史"""
        self._turns = []
        self._save()
    
    def get_all_turns(self) -> List[ConversationTurn]:
        """获取所有对话轮次"""
        return self._turns.copy()
    
    @property
    def user_id(self) -> str:
        """获取当前用户ID"""
        return self._user_id
    
    @user_id.setter
    def user_id(self, value: str) -> None:
        """设置用户ID（私有属性，通过构造函数设置）"""
        self._user_id = value