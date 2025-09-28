# llm_framework/main.py
import os
import sys
import json
from dotenv import load_dotenv
from llm_orchestrator import LLMOrchestrator
# 引入策略类
from selection_strategy import SequentialStrategy, RandomStrategy 
from tqdm import tqdm
from manyllm import ChatSession

# --- 定义模型优先级列表 ---
MODEL_PRIORITY_LIST = [
    "gemini-2.0-flash-lite",
    # "gpt-4o",  # 可以把 OpenAI 作为备用
    "gemini-1.5-flash-latest",
]

system_prompt = """
## 角色和目标：
你是一位专业的HR专家和语言模型，专门负责撰写高质量、专业且无歧视的职位描述（Job Description）。你的核心任务是根据用户提供的职位名称和基本要求，生成一份结构清晰、内容详尽、完全合规的岗位说明。

## 核心指令：

#### 分析输入： 仔细分析用户输入，提取职位名称、学历要求、经验年限等核心要素。

#### 内容生成：

- 岗位职责 (Job Responsibilities): 生成3-5条具体、明确、以动词开头的核心工作职责，清晰描述该职位的日常工作内容和主要目标。

- 任职要求 (Job Requirements): 必须分为三个明确的子类进行描述：

- 基本要求 (Basic Requirements): 严格根据用户提供的学历和经验年限信息进行陈述。

- 专业技能 (Professional Skills): 列出胜任该职位所必需的硬技能，例如：编程语言、技术框架、理论知识、熟悉的工具软件等。

- 软技能 (Soft Skills): 列出重要的非技术性能力，例如：沟通能力、团队协作精神、问题解决能力、逻辑思维、学习能力等。

#### 严格遵守反歧视原则：

- 你必须严格遵守劳动法规和全球最佳招聘实践，严禁在输出的任何部分包含歧视性语言或倾向。

- 严禁出现任何基于性别、年龄、种族、民族、宗教信仰、婚姻状况、户籍所在地等非专业能力因素的限制性或偏好性描述。

- 绝对禁止使用例如“男性优先”、“年龄要求35岁以下”、“仅限本地户籍”、“身体健康，无不良嗜好”等歧视性或无关表述。所有要求都必须与岗位胜任能力直接相关。

- 禁止出现公司名，公司背景，公司介绍等信息。

#### 格式化输出：

- 你的最终输出必须是纯文本的Markdown格式。

- 所有换行必须使用 \n 字符来表示。

- 必须严格遵循下面提供的结构模板，不得有任何格式偏差。

```### [在这里填写职位名称]\n\n**岗位职责：**\n1. [在这里填写具体、以动词开头的职责描述1]\n2. [在这里填写具体、以动词开头的职责描述2]\n3. [在这里填写具体、以动词开头的职责描述3]\n\n**任职要求：**\n**1. 基本要求：**\n   * 学历：[根据用户输入填写学历]\n   * 经验：[根据用户输入填写经验年限]\n\n**2. 专业技能：**\n   * [在这里填写必需的专业技能1]\n   * [在这里填写必需的专业技能2]\n   * [在这里填写必需的专业技能3]\n\n**3. 软技能：**\n   * [在这里填写相关的软技能1]\n   * [在这里填写相关的软技能2]```

"""

user_msg = """
{"messages": [{"role": "user", "content": "如果我要招聘一个嵌入式开发工程师，要求是本科和不限，职位描述应该怎么写？"}, {"role": "assistant", "content": "为您生成了一份嵌入式开发工程师的招聘详情：\n\n**我们正在寻找一位嵌入式开发工程师！**\n* **我们希望您拥有：** 本科学历以及不限的相关工作经验。\n* **您将负责：**\n工作职责;1.负责车控操作OS 下功能模块设计与软件开发工作；2.负责车控操作OS 下 功能模块的功能定义、规范定义、模块的软件架构设计；任职资格：1.车辆工程、通信、计算机、软件工程、交通等专业，本科及以上学历；2.精通嵌入式Linux软件开发及调试技巧，熟练掌握C、C++编程语言之一及多线程编程或网络编程；3.熟悉无人驾驶或机器人系统功能协议栈，有项目开发经验优先；4.具备良好的分析解决问题能力，能独立承担开发任务和有系统进度把控能力；5.责任心强，良好的对外沟通和团队协作能力，主动，好学；背景介绍：国汽智控在去年7月31日正式挂牌成立，秉承着顶层设计和行业共识，是由国汽智联孵化的面向市场的公司，近期天使轮融资近一个亿；安全部门由ICT行业和车圈资深研发人员组成，部门研发氛围浓厚，同事关系简单融洽，彼此相互沟通，相互学习；部门leader是清华自动化系的本科硕士，博士毕业于伊利诺伊大学香槟分校电子与计算机工程系，具备20多年的信息安全和云计算经验；公司创始人尚进是清华汽车系的本科、硕士、博士，并获计算机本科双学位，博士论文获评“全国百篇优秀博士论文”。2001年获美国南加州大学计算机硕士学位，在ICT和汽车行业均有很资深的从业背景；"}]}
"""



# 原有的 run_chat_session 函数可以保留，用于测试非流式

def run_streaming_chat_session():
    """
    演示流式调用的函数。
    """
    load_dotenv()
    orchestrator = LLMOrchestrator(model_priority=MODEL_PRIORITY_LIST)
    
    messages = [
        {"role": "user", "content": user_msg}
    ]
    print(f"用户: {messages[-1]['content']}\n")
    
    # --- 调用流式接口 ---
    stream_generator = orchestrator.chat_stream(messages)
    
    print("🤖 模型回复 (流式):")
    full_response = ""
    for chunk in stream_generator:
        # 实时打印每个文本块
        print(chunk, end="")
        sys.stdout.flush() # 确保立即输出到控制台
        full_response += chunk
    
    print("\n\n--- 流式传输结束 ---")
    # print(f"完整回复: \n{full_response}")


if __name__ == "__main__":
    # print("--- 运行流式聊天会话 ---")
    # run_streaming_chat_session()
    
    # 你也可以保留非流式测试
    print("\n\n--- 运行非流式聊天会话 ---")
    # from main import run_chat_session # 假设你把原来的代码放到了同名函数
    chat = ChatSession()

    chat.run_chat(user_msg, system_prompt=system_prompt)
