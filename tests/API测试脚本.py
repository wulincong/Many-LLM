# -*- coding: utf-8 -*-
"""
Youcai 模型 API Python 调用示例

使用说明：
1. 在下方填写 API_KEY（如需鉴权）
2. 运行：python call_youcai_mini.py
"""

import requests

# ============================================================
# 👇 在这里填写你的配置
# ============================================================
API_KEY = "gw-kamci80pL1UO4TEk43FZR5IiWvm_RRE5IT7pNmXqP6s"  # 如果后端需要鉴权，填在这里
# ============================================================

BASE_URL = "http://8.134.216.249:19111"


def chat(question: str) -> str:
    """非流式对话"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "youcai",
            "messages": [
                {"role": "user", "content": question},
            ],
        },
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "你好，请做一个自我介绍"
    print(chat(question))
