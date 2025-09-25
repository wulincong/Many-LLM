import os
import sys
import json
from dotenv import load_dotenv
from llm_orchestrator import LLMOrchestrator
from tqdm import tqdm # 引入tqdm来显示进度条，需要 pip install tqdm
import time # 引入time模块用于演示

# --- 定义模型优先级列表 ---
MODEL_PRIORITY_LIST = [
    # 注意：Gemini API中的模型名称通常不含 '2.0-flash-lite'，请确认您拥有访问权限的模型名称
    # 常用的模型是 "gemini-1.5-pro-latest" 或 "gemini-1.5-flash-latest"
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash"
    # "gpt-4o",
]

# --- 角色设定与指令 (System Prompt) ---
# 您的 system_prompt 非常出色，我们直接使用
SYSTEM_PROMPT = """
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

def create_optimization_prompt(original_user_content, original_assistant_content):
    """根据原始数据，构建用于优化的新Prompt"""
    return f"""
请严格遵循您作为HR专家的角色和准则，对以下职位描述草稿进行优化和重写。

# 原始用户请求:
{original_user_content}

# 待优化的助理草稿:
{original_assistant_content}

# 你的任务:
请忽略草稿中所有不合规（如公司介绍、歧视性语言）和格式不正确的内容，重新生成一份完全符合您角色设定中所有指令的、专业的、结构化的职位描述。
"""
def process_dataset_file(input_path, output_path):
    """
    读取jsonl文件，调用LLM进行优化，并将结果写入新的jsonl文件。
    支持实时写入和断点续传。
    """
    load_dotenv()
    orchestrator = LLMOrchestrator(model_priority=MODEL_PRIORITY_LIST)

    # --- 1. 断点续传：检查已完成的记录数 ---
    processed_count = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                # 高效地计算文件行数
                processed_count = sum(1 for _ in f)
            print(f"检测到已存在的输出文件，包含 {processed_count} 条记录。将从断点处继续...")
        except Exception as e:
            print(f"警告：无法读取输出文件 '{output_path}' 的行数，将从头开始。错误: {e}")
    
    # --- 2. 断点续传：决定文件打开模式 ---
    # 如果已有记录，则以追加模式('a')打开；否则以写入模式('w')创建新文件
    file_mode = 'a' if processed_count > 0 else 'w'
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            # 为了准确显示tqdm进度，先获取总行数
            total_lines = sum(1 for _ in infile)
            # 重置文件指针以便再次读取
            infile.seek(0)
            
            print(f"输入文件 '{input_path}' 加载成功，共 {total_lines} 条记录。")

            # --- 3. 断点续传：跳过已处理的输入行 ---
            for _ in range(processed_count):
                next(infile, None)

            # 使用追加模式打开输出文件
            with open(output_path, file_mode, encoding='utf-8') as outfile:
                # 初始化进度条
                pbar = tqdm(
                    enumerate(infile, start=processed_count + 1), 
                    total=total_lines, 
                    initial=processed_count, 
                    desc="处理进度"
                )
                
                for line_num, line in pbar:
                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        
                        user_msg = next((m for m in messages if m['role'] == 'user'), None)
                        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)

                        if not user_msg or not assistant_msg:
                            print(f"警告: 输入文件第 {line_num} 行数据格式不完整，已跳过。")
                            continue

                        optimization_prompt = create_optimization_prompt(
                            user_msg['content'], 
                            assistant_msg['content']
                        )
                        llm_messages = [{"role": "user", "content": optimization_prompt}]

                        result = orchestrator.chat(
                            messages=llm_messages,
                            temperature=0.3,
                            max_tokens=2048,
                            system_prompt=SYSTEM_PROMPT
                        )

                        output_record = {
                            "line_number": line_num,
                            "original_user_request": user_msg['content'],
                            "original_assistant_response": assistant_msg['content'],
                        }

                        if result["status"] == "success":
                            output_record["optimized_response"] = result['content']
                            output_record["processed_by_model"] = result['model']
                        else:
                            output_record["optimized_response"] = f"ERROR: {result['message']}"
                            output_record["processed_by_model"] = "None"
                        
                        outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                        
                        # --- 4. 实时写入：强制将缓冲区内容写入磁盘 ---
                        outfile.flush()

                    except json.JSONDecodeError:
                        print(f"警告: 第 {line_num} 行不是有效的JSON，已跳过。")
                    except Exception as e:
                        print(f"处理第 {line_num} 行时发生未知错误: {e}")
                        # 可以在这里选择是停止还是继续
                        time.sleep(1)

    except FileNotFoundError:
        print(f"错误: 输入文件未找到于 '{input_path}'")
        sys.exit(1)
        
    print(f"\n处理完成！所有结果已保存到 '{output_path}'。")


if __name__ == "__main__":
    INPUT_FILE = "dataset/智能网联汽车.jsonl"
    OUTPUT_FILE = "dataset/optimized_jds.jsonl"
    
    print("--- 开始批量优化职位描述文件 ---")
    process_dataset_file(input_path=INPUT_FILE, output_path=OUTPUT_FILE)