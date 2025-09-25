# llm_framework/llm_factory.py
import os
from typing import Optional
from providers.base_provider import LLMProvider
from providers.gemini_provider import GeminiProvider
# 当你添加新的模型时，在这里导入
# from providers.openai_provider import OpenAIProvider 

class LLMFactory:
    @staticmethod
    def get_provider(model_name: str, **kwargs) -> Optional[LLMProvider]:
        """
        根据模型名称获取相应的 LLM 提供者实例。
        :param model_name: 模型全名, e.g., "gemini-1.5-pro"
        :return: LLMProvider 的实例, 如果不支持则返回 None
        """
        if "gemini" in model_name.lower():
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("请设置 GOOGLE_API_KEY 环境变量")
            return GeminiProvider(model_name=model_name, api_key=api_key, **kwargs)
        
        # --- 扩展点：在这里添加对其他模型的支持 ---
        # elif "gpt" in model_name.lower():
        #     api_key = os.getenv("OPENAI_API_KEY")
        #     if not api_key:
        #         raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        #     return OpenAIProvider(model_name=model_name, api_key=api_key)
        
        else:
            raise ValueError(f"不支持的模型: {model_name}")