"""路径初始化工具：确保项目根目录在 sys.path 中"""

import sys
from pathlib import Path


def setup_project_path() -> Path:
    """
    将项目根目录添加到 sys.path，并返回根目录 Path 对象。
    
    调用示例：
        from src.utils.path_setup import setup_project_path
        project_root = setup_project_path()
    """
    # 获取项目根目录（假设此文件在 src/utils/ 下）
    project_root = Path(__file__).parent.parent.parent
    project_root_str = str(project_root)
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root