"""选择 Llama3/Qwen、模型路径、max_tokens 等"""

from pathlib import Path
from enum import Enum
from typing import Optional


class LLMBackendType(str, Enum):
    """LLM 后端类型"""
    HF_TRANSFORMERS = "hf_transformers"  # transformers + bitsandbytes
    LLAMA_CPP = "llama_cpp"  # llama-cpp-python


class LLMConfig:
    """LLM 配置"""
    
    # 后端选择
    backend_type: LLMBackendType = LLMBackendType.HF_TRANSFORMERS
    
    # 模型路径（Hugging Face 模型名称或本地路径）
    # 示例：
    # - "Qwen/Qwen2.5-7B-Instruct" (Hugging Face)
    # - "meta-llama/Llama-3-8B-Instruct" (Hugging Face)
    # - "/path/to/local/model" (本地路径)
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 量化配置
    load_in_4bit: bool = True  # 4-bit 量化
    load_in_8bit: bool = False  # 8-bit 量化（与4-bit互斥）
    
    # 生成参数
    max_new_tokens: int = 512  # 最大生成token数
    temperature: float = 0.7  # 温度参数
    top_p: float = 0.9  # nucleus sampling
    top_k: int = 50  # top-k sampling
    do_sample: bool = False  # 是否使用采样
    
    # 设备配置
    device_map: str = "auto"  # 自动分配设备
    
    # llama.cpp 特定配置
    llama_cpp_model_path: Optional[str] = None  # llama.cpp 模型文件路径（.gguf格式）
    llama_cpp_n_ctx: int = 4096  # 上下文窗口大小
    llama_cpp_n_threads: int = 4  # CPU线程数
    llama_cpp_n_gpu_layers: int = 35  # GPU层数（-1表示全部使用GPU）
    
    # 其他配置
    trust_remote_code: bool = True  # 是否信任远程代码（某些模型需要）
    use_cache: bool = True  # 是否使用KV缓存
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "LLMConfig":
        """从字典创建配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config