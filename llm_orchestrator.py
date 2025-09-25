# llm_framework/llm_orchestrator.py
# ... __init__ 和 chat 方法不变 ...
# llm_framework/llm_orchestrator.py
from typing import List, Dict, Any
from llm_factory import LLMFactory

class LLMOrchestrator:
    """
    模型编排器，负责根据优先级列表调用模型，并处理故障切换。
    """
    def __init__(self, model_priority: List[str]):
        """
        初始化编排器。
        :param model_priority: 一个包含模型名称的列表，按从高到低的优先级排列。
        """
        if not model_priority:
            raise ValueError("模型优先级列表不能为空")
        self.model_priority = model_priority
        self.factory = LLMFactory()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        尝试按顺序调用模型列表进行对话。
        :param messages: 对话历史。
        :param kwargs: 其他生成参数。
        :return: 一个包含成功模型和其回复的字典，或者一个错误信息。
        """
        last_exception = None
        for model_name in self.model_priority:
            try:
                print(f"--- 尝试使用模型: {model_name} ---")
                provider = self.factory.get_provider(model_name, **kwargs)
                response_text = provider.chat(messages, **kwargs)
                print(f"--- 模型 {model_name} 调用成功！ ---")
                # 返回一个结构化的结果，告知哪个模型成功了
                return {
                    "status": "success",
                    "model": model_name,
                    "content": response_text
                }
            except Exception as e:
                print(f"模型 {model_name} 调用失败。错误: {e}")
                last_exception = e
                # 继续尝试下一个模型
                continue
        
        # 如果所有模型都失败了
        print("--- 所有模型均调用失败 ---")
        return {
            "status": "error",
            "message": f"所有模型都无法处理请求。最后一个错误: {last_exception}"
        }
    
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        """
        尝试按顺序调用模型列表进行流式对话。
        """
        last_exception = None
        for model_name in self.model_priority:
            try:
                print(f"--- 尝试使用模型 (流式): {model_name} ---")
                provider = self.factory.get_provider(model_name)
                # yield from 会将子生成器的产出直接传递给调用者
                yield from provider.chat_stream(messages, **kwargs)
                
                print(f"\n--- 模型 {model_name} 流式传输成功！ ---")
                # 如果成功启动流，就此结束
                return
            except Exception as e:
                print(f"模型 {model_name} (流式) 调用失败。错误: {e}")
                last_exception = e
                continue
        
        # 如果所有模型都失败了
        print("--- 所有模型均调用失败 (流式) ---")
        # 我们可以 yield 一个错误信息，或者直接抛出异常
        yield f"所有模型都无法处理请求。最后一个错误: {last_exception}"