"""基于 llama-cpp-python"""

from typing import Optional

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

from src.llm.backends.config import LLMConfig


class LlamaCppBackend:
    """基于 llama-cpp-python 的 LLM 后端"""
    
    def __init__(self, config: LLMConfig):
        """
        初始化后端
        
        :param config: LLM 配置
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
            )
        
        self.config = config
        self.llm: Optional[Llama] = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """延迟初始化模型（按需加载）"""
        if self._initialized:
            return
        
        model_path = self.config.llama_cpp_model_path or self.config.model_name_or_path
        
        if not model_path.endswith('.gguf'):
            raise ValueError("llama.cpp 后端需要 .gguf 格式的模型文件")
        
        print(f"正在加载模型: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.config.llama_cpp_n_ctx,
            n_threads=self.config.llama_cpp_n_threads,
            n_gpu_layers=self.config.llama_cpp_n_gpu_layers,
            verbose=False
        )
        
        self._initialized = True
        print("模型加载完成")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        生成回复
        
        :param prompt: 输入提示词
        :param max_new_tokens: 最大生成token数
        :param temperature: 温度参数
        :param top_p: nucleus sampling
        :param top_k: top-k sampling
        :return: 生成的文本
        """
        if not self._initialized:
            self._initialize()
        
        # 使用配置中的默认值
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        
        # 生成
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
            stop=["\n\n", "用户：", "助手："]  # 停止词
        )
        
        generated_text = output["choices"][0]["text"].strip()
        return generated_text
    
    def unload(self) -> None:
        """卸载模型（释放显存）"""
        if self.llm is not None:
            del self.llm
            self.llm = None
        self._initialized = False
        print("模型已卸载")