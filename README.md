# Many-LLM

本项目旨在尝试使用多种（免费）大模型回答问题，并通过策略进行负载均衡和故障切换。

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
