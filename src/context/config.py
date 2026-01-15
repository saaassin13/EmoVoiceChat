"""最多保留轮数（如 10）、文件路径"""

from pathlib import Path

# 对话历史配置
MAX_HISTORY_TURNS = 10  # 最多保留的对话轮数
HISTORY_DIR = Path("data/history")  # 对话历史目录（每个用户一个文件）