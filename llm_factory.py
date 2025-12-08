# llm_framework/llm_factory.py (重构后)
from typing import Optional
from providers.base_provider import LLMProvider
from providers.gemini_provider import GeminiProvider
from providers.zhipuai_provider import ZhipuAIProvider
# from providers.openai_provider import OpenAIProvider

class LLMFactory:
    @staticmethod
    def get_provider(model_name: str, api_key: str) -> Optional[LLMProvider]:
        """
        根据模型名称和传入的API密钥获取相应的LLM提供者实例。
        """
        if not api_key:
            raise ValueError("API Key 不能为空")

        if "gemini" in model_name.lower() or "gemma" in model_name.lower():
            return GeminiProvider(model_name=model_name, api_key=api_key)
        elif "glm" in model_name.lower():
            return ZhipuAIProvider(model_name=model_name, api_key=api_key)
        # elif "gpt" in model_name.lower():
        #     return OpenAIProvider(model_name=model_name, api_key=api_key)
        
        else:
            raise ValueError(f"不支持的模型或未在工厂中注册: {model_name}")