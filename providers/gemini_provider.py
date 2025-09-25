# llm_framework/providers/gemini_provider.py
from google import genai
from typing import List, Dict, Generator
from .base_provider import LLMProvider
from google.genai import types

class GeminiProvider(LLMProvider):
    """
    Google Gemini 模型的具体实现，使用 genai.Client。
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        # 配置 API 密钥
        extra_args = {}
        if "system_prompt" in kwargs.keys():
            extra_args["system_instruction"] = kwargs["system_prompt"]
        if "thinking" in kwargs.keys():
            extra_args["thinking_config"] = types.ThinkingConfig(thinking_budget=kwargs["thinking"])
        
        self.generation_config = types.GenerateContentConfig(**extra_args)
        
        self.client = genai.Client(api_key=self.api_key)
        
    def _prepare_contents(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """
        私有辅助函数，将我们的标准消息格式转换为 Gemini 的内容格式。
        """
        contents = []
        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else msg['role']
            contents.append({'role': role, 'parts': [{'text': msg['content']}]})
        return contents

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        非流式聊天实现。
        """
        try:
            contents = self._prepare_contents(messages)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config = self.generation_config
            )

            return response.text
        except Exception as e:
            print(f"调用 Gemini API (非流式) 时出错: {e}")
            raise e

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        流式聊天实现。
        """
        try:
            contents = self._prepare_contents(messages)
            generation_config = {
                'temperature': kwargs.get('temperature', 0.7),
                'max_output_tokens': kwargs.get('max_tokens', 1024)
            }
            # contents = [_["content"] for _ in messages]
            # 调用流式接口
            stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"调用 Gemini API (流式) 时出错: {e}")
            raise e

