import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StopStringCriteria,
    StoppingCriteriaList,
    pipeline
)
from typing import Optional

from src.llm.backends.config import LLMConfig


class HFTransformersBackend:
    """基于 transformers + bitsandbytes 的 LLM 后端"""

    def __init__(self, config: LLMConfig):
        """
        初始化后端

        :param config: LLM 配置
        """
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = False

    def _initialize(self) -> None:
        """延迟初始化模型（按需加载）"""
        if self._initialized:
            return

        print(f"正在加载模型: {self.config.model_name_or_path}")

        # 配置量化
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code
        )

        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型（缓存到 ~/.cache/huggingface/hub/）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quantization_config,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.float16 if not quantization_config else None,
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
        基于纯字符串 prompt 的生成（兼容旧接口）

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

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode（只取新生成部分）
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def generate_from_messages(
        self,
        messages: list,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        基于 chat messages + apply_chat_template 的生成（适配 Qwen 等聊天模型）

        :param messages: 形如 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
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

        # 1) 使用 chat_template 构造 prompt
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2) tokenize
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

        # 3) generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 4) decode（只取新生成部分）
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def unload(self) -> None:
        """卸载模型（释放显存）"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        self._initialized = False
        print("模型已卸载")