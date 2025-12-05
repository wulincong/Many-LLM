import os
import sys
import json
from dotenv import load_dotenv
from llm_orchestrator import LLMOrchestrator
# 引入策略类
from selection_strategy import SequentialStrategy, RandomStrategy 
from tqdm import tqdm


class ChatSession:
    def __init__(self):
        load_dotenv()
        self.load_model_key_pool_from_env()
        self.strategy = RandomStrategy()    # 或者使用随机策略进行负载均衡

    def load_model_key_pool_from_env(self):
        """从环境变量加载并构建 (模型, 密钥) 池"""
        self.pool = []
        
        # 加载 Gemini 密钥
        for i in range(1, 10): # 最多检查9个key
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                # 为池中的每个项添加元数据，例如优先级
                self.pool.append(("gemma-3-27b-it", key, {"priority": 1, "provider": "google"}))
                self.pool.append(("gemini-2.0-flash-lite", key, {"priority": 2, "provider": "google"}))
                self.pool.append(("gemini-2.0-flash", key, {"priority": 3, "provider": "google"}))
        
        # 加载 OpenAI 密钥
        for i in range(1, 10):
            key = os.getenv(f"OPENAI_API_KEY_{i}")
            if key:
                self.pool.append(("gpt-4o", key, {"priority": 3, "provider": "openai"}))

        if not self.pool:
            raise ValueError("未能从环境变量加载任何 (模型,密钥) 对，请检查 .env 文件。")
            
        # 根据元数据中的优先级对池进行排序
        self.pool.sort(key=lambda item: item[2].get("priority", 99))
        
        print("--- 模型密钥池加载完成 ---")
        for model, key, meta in self.pool:
            print(f"  - 模型: {model}, 元数据: {meta}, Key: ...{key[-4:]}")
        print("--------------------------")


    def run_chat(self, user_msg, system_prompt="", temperature=0.7, max_tokens=1000):
        
        print(f"当前使用的选择策略: {self.strategy.__class__.__name__}")

        # 3. 初始化编排器
        orchestrator = LLMOrchestrator(pool=self.pool, strategy=self.strategy)

        messages = [
            {"role": "user", "content": user_msg}
        ]

        # print(f"用户: {messages[-1]['content']}\n")

        # --- 使用编排器进行调用 ---
        result = orchestrator.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=0,
            system_prompt = system_prompt
        )

        # --- 处理返回结果 ---
        if result["status"] == "success":
            # print(f"\n✅ 最终成功模型: {result['model']}")
            return result['content']
        else:
            # print(f"\n❌ 请求失败:")
            # print(result["message"])
            return result["message"]

if __name__ == "__main__":
    chat = ChatSession()
    chat.run_chat("什么是量子力学")