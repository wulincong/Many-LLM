# Many-LLM

## 使用场景

* 想 **大批量“白嫖”LLM** —— 想让LLM 7×24 小时给你打工，又不愿意支付长长的账单。
* 需要跑一大堆**数据清洗、文本处理、文献翻译**，但你本人拒绝加班，希望 AI 帮你 996。
* 想把 **LLM 接入自己的程序（LLM-in-the-loop）**，结果发现大模型们不是在限流，就是在罢工，调用成功率堪比摇号。

本项目的目标，是整合多个免费可用的大语言模型源，通过智能调度、负载均衡与自动故障切换机制，让你的任务在模型波动频繁的情况下依旧稳定运行，实现“有模型可用就用、没模型就自动切换”的弹性策略。


## 快速开始

### 1. 配置环境变量 (.env)

在项目根目录下创建一个名为 `.env` 的文件，并按照以下格式配置您的大模型 API 密钥。本项目支持 Google Gemini 和 Zhipu 模型。

**Google Gemini API 密钥配置示例：**

```dotenv
GEMINI_API_KEY_1=YOUR_GEMINI_API_KEY_1
GEMINI_API_KEY_2=YOUR_GEMINI_API_KEY_2
# 您可以根据需要添加更多 Gemini API 密钥
```

**OpenAI API 密钥配置示例：**

```dotenv
OPENAI_API_KEY_1=YOUR_OPENAI_API_KEY_1
# 您可以根据需要添加更多 OpenAI API 密钥
```

**Cloudflare AI Gateway API 密钥配置示例：**

```dotenv
CLOUDFLARE_API_BASE_URL=https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/ai/run/
CLOUDFLARE_API_TOKEN=YOUR_CLOUDFLARE_API_TOKEN
# 请将 YOUR_ACCOUNT_ID 替换为您的 Cloudflare 账户 ID
# 请将 YOUR_CLOUDFLARE_API_TOKEN 替换为您的 Cloudflare API Token
```

**重要提示：**
*   请确保您的 API 密钥是有效的。
*   `manyllm.py` 会自动从 `.env` 文件加载这些密钥，并构建模型池。
*   如果未配置任何有效的 (模型, 密钥) 对，程序将抛出 `ValueError`。

### 2. 运行项目

您可以通过运行 `main.py` 或 `manyllm.py` 来启动聊天会话。

**使用 `main.py` (示例用途，包含特定系统提示和用户消息):**

```bash
python main.py
```

**使用 `manyllm.py` (更通用的聊天会话):**

```bash
python manyllm.py
```

### 3. 模型选择策略

本项目支持两种模型选择策略：

*   **`SequentialStrategy` (顺序策略):** 按照模型池中定义的优先级顺序依次尝试调用模型。
*   **`RandomStrategy` (随机策略):** 随机选择模型进行调用，实现负载均衡。

您可以在 `manyllm.py` 中修改 `ChatSession` 的 `strategy` 属性来切换策略：

```python
# manyllm.py
# ...
from selection_strategy import SequentialStrategy, RandomStrategy 

class ChatSession:
    def __init__(self):
        # ...
        self.strategy = RandomStrategy()    # 或者使用 SequentialStrategy()
        # ...
```

### 4. 项目结构概览

*   `llm_factory.py`: 负责根据模型名称和 API 密钥创建相应的 LLM 提供者实例。
*   `llm_orchestrator.py`: 模型编排器，根据选择策略管理模型调用和故障切换。
*   `manyllm.py`: 核心聊天会话逻辑，加载环境变量，初始化模型池和策略。
*   `main.py`: 包含一个使用特定系统提示和用户消息的示例运行。
*   `selection_strategy.py`: 定义了不同的模型选择策略。
*   `providers/`: 包含不同大模型提供者的实现 (例如 `gemini_provider.py`, `openai_provider.py`)。
*   `dataset/`: 存放数据集文件。
*   `utils/`: 存放工具函数和 Jupyter Notebook。

## TODO:

- 以API模式启动
- 并行批处理任务支持
- 以指定编程语言的格式返回，而不是仅返回字符串

