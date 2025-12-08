# llm_framework/providers/zhipuai_provider.py
from zai import ZhipuAiClient
from typing import List, Dict, Generator
from .base_provider import LLMProvider

class ZhipuAIProvider(LLMProvider):
    """
    ZhipuAI GLM 模型的具体实现，使用 ZhipuAiClient。
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.client = ZhipuAiClient(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        非流式聊天实现。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                thinking={
                    "type": "disabled",
                },
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.6)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用 ZhipuAI API (非流式) 时出错: {e}")
            raise e

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        流式聊天实现。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                thinking={
                    "type": "disabled",
                },
                stream=True,
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.6)
            )

            for chunk in response:
                if chunk.choices[0].delta.reasoning_content:
                    yield chunk.choices[0].delta.reasoning_content
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"调用 ZhipuAI API (流式) 时出错: {e}")
            raise e