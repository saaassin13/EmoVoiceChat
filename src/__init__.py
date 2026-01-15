"""项目初始化：将项目根目录添加到 sys.path"""

import sys
from pathlib import Path

# 获取项目根目录（src 的父目录）
_project_root = Path(__file__).parent.parent
_project_root_str = str(_project_root)

# 如果不在 sys.path 中，则添加
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)