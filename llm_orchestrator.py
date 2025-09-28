# llm_framework/llm_orchestrator.py
# ... __init__ 和 chat 方法不变 ...
# llm_framework/llm_orchestrator.py
from typing import List, Dict, Any, Generator
from llm_factory import LLMFactory
from selection_strategy import SelectionStrategy, PoolItem, ItemIdentifier

class LLMOrchestrator:
    """
    模型编排器，负责根据优先级列表调用模型，并处理故障切换。
    """
    def __init__(self, pool: List[PoolItem], strategy: SelectionStrategy):
        if not pool:
            raise ValueError("模型密钥池不能为空")
        self.pool = pool
        self.strategy = strategy
        self.factory = LLMFactory()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        尝试按顺序调用模型列表进行对话。
        :param messages: 对话历史。
        :param kwargs: 其他生成参数。
        :return: 一个包含成功模型和其回复的字典，或者一个错误信息。
        """
        failed_items: set[ItemIdentifier] = set()
        last_exception = None

        while True:
            # 向策略请求下一个要尝试的项
            selected_item = self.strategy.select(self.pool, failed_items)

            # 如果策略返回None，说明所有选项都已尝试失败
            if selected_item is None:
                break
            
            model_name, api_key, metadata = selected_item
            
            try:
                print(f"--- 策略选择: 模型={model_name}, 元数据={metadata}, Key=...{api_key[-4:]} ---")
                provider = self.factory.get_provider(model_name, api_key)
                response_text = provider.chat(messages, **kwargs)
                
                print(f"--- 调用成功! ---")
                return {
                    "status": "success",
                    "model": model_name,
                    "key_used": f"...{api_key[-4:]}",
                    "metadata": metadata,
                    "content": response_text
                }
            except Exception as e:
                print(f"--- 调用失败。错误: {e} ---")
                last_exception = e
                # 将失败的项加入集合，以便策略下次选择时跳过
                failed_items.add((model_name, api_key))
        
        print("--- 所有可用 (模型,密钥) 对均调用失败 ---")
        return {
            "status": "error",
            "message": f"所有可用选项都无法处理请求。最后一个错误: {last_exception}"
        }

    # chat_stream 方法也需要进行同样的修改
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        failed_items: set[ItemIdentifier] = set()
        last_exception = None

        while True:
            selected_item = self.strategy.select(self.pool, failed_items)
            if selected_item is None:
                break

            model_name, api_key, metadata = selected_item
            try:
                print(f"--- 策略选择 (流式): 模型={model_name}, 元数据={metadata}, Key=...{api_key[-4:]} ---")
                provider = self.factory.get_provider(model_name, api_key)
                yield from provider.chat_stream(messages, **kwargs)
                print(f"\n--- 模型 {model_name} 流式传输成功！ ---")
                return
            except Exception as e:
                print(f"--- 调用失败 (流式)。错误: {e} ---")
                last_exception = e
                # --- 核心修改 ---
                failed_items.add((model_name, api_key))

        print("--- 所有可用 (模型,密钥) 对均调用失败 (流式) ---")
        yield f"ERROR: 所有可用选项都无法处理请求。最后一个错误: {last_exception}"

