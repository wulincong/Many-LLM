# llm_framework/providers/base_provider.py
from abc import ABC, abstractmethod
from typing import List, Dict, Generator

class LLMProvider(ABC):
    """
    大语言模型提供者的抽象基类 (统一接口).
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        初始化模型提供者。
        :param model_name: 具体的模型名称, e.g., "gemini-1.5-pro-latest"
        :param api_key: 该模型对应的 API Key
        """
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        进行多轮对话。
        :param messages: 对话历史, 格式为 [{"role": "user/model", "content": "..."}]
        :param kwargs: 其他特定于模型的参数 (e.g., temperature, max_tokens)
        :return: 模型生成的回复文本
        """
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """
        以流式方式进行多轮对话。
        :param messages: 对话历史
        :param kwargs: 其他参数
        :return: 一个返回文本块(chunk)的生成器
        """
        # 实现这个方法的类应该使用 'yield' 关键字
        yield ""


    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"