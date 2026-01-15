"""统一一个 generate_reply(text, emotion, history) 接口"""

from typing import List, Optional

from src.llm.backends.config import LLMConfig, LLMBackendType
from src.llm.backends.hf_transformers_backend import HFTransformersBackend
from src.llm.backends.llama_cpp_backend import LlamaCppBackend
from src.llm.prompt_builder import PromptBuilder
from src.context.schemas import ConversationTurn
from src.ser.schemas import EmotionLabel


class LLMEngine:
    """统一的 LLM 引擎接口"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 引擎
        
        :param config: LLM 配置，None 则使用默认配置
        """
        self.config = config or LLMConfig()
        self.backend = self._create_backend()
        self.prompt_builder = PromptBuilder()
    
    def _create_backend(self):
        """根据配置创建对应的后端"""
        if self.config.backend_type == LLMBackendType.HF_TRANSFORMERS:
            return HFTransformersBackend(self.config)
        elif self.config.backend_type == LLMBackendType.LLAMA_CPP:
            return LlamaCppBackend(self.config)
        else:
            raise ValueError(f"不支持的后端类型: {self.config.backend_type}")
            
    def generate_reply(
        self,
        user_text: str,
        user_emotion: EmotionLabel,
        history: List[ConversationTurn],
        **generation_kwargs
    ) -> str:
        """
        生成回复（统一接口）
        """
        # 这里只对 HF_TRANSFORMERS 后端生效；如果未来有别的后端，可以做判断
        if isinstance(self.backend, HFTransformersBackend):
            messages = self.prompt_builder.build_chat_messages(
                user_text=user_text,
                user_emotion=user_emotion,
                history=history
            )
            reply = self.backend.generate_from_messages(messages, **generation_kwargs)
        else:
            # 回退到旧的纯字符串 prompt（兼容其它后端）
            prompt = self.prompt_builder.build_prompt(
                user_text=user_text,
                user_emotion=user_emotion,
                history=history
            )
            reply = self.backend.generate(prompt, **generation_kwargs)

        # 后处理

        return reply
    
    def unload(self) -> None:
        """卸载模型（释放显存）"""
        self.backend.unload()