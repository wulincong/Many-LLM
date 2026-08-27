import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable, Iterable


RESULT_COLUMN = "\u5224\u522b\u7ed3\u679c"
NON_SENSITIVE_MARK = "\u4e0d\u5c5e\u4e8e\u654f\u611f\u8bcd"
MANUAL_REVIEW_MARK = "None"
DEFAULT_KEYWORD_COLUMN = "\u5173\u952e\u8bcd"
DEFAULT_INPUT = Path("dataset") / "附件4-拦截关键词0727.csv"
DEFAULT_OUTPUT = Path("dataset") / "sensitive_keyword_model_review.csv"


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0], rows[1:]


def parse_non_sensitive_terms(text: str | None) -> list[str]:
    if text is None:
        raise ValueError("empty model response: got None")
    text = text.strip()
    if not text:
        raise ValueError("empty model response")

    match = re.search(r"\[[\s\S]*\]", text)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    terms = []
    for line in text.splitlines():
        line = line.strip().strip(",;")
        line = re.sub(r"^\s*(?:[-*]|\d+[.)]|[\"'])\s*", "", line).strip()
        line = line.strip("\"'`")
        if line and line not in {"[]", "\u65e0", "none", "None", "NULL"}:
            terms.append(line)
    return terms


def resolve_keyword_column(header: list[str], keyword_column: str | None) -> int | None:
    if keyword_column is None:
        if DEFAULT_KEYWORD_COLUMN in header:
            return header.index(DEFAULT_KEYWORD_COLUMN)
        return None
    if keyword_column.isdigit():
        index = int(keyword_column)
        if index < 0:
            raise ValueError("--keyword-column index must be >= 0")
        return index
    if keyword_column not in header:
        raise ValueError(f"Column not found: {keyword_column}")
    return header.index(keyword_column)


def keyword_from_row(row: list[str], column_index: int | None) -> str:
    if column_index is not None:
        return row[column_index].strip() if column_index < len(row) else ""
    return next((cell.strip() for cell in row if cell.strip()), "")


def write_output(path: Path, header: list[str], rows: list[list[str]], results: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_header = list(header)
    if not output_header or output_header[-1] != RESULT_COLUMN:
        output_header.append(RESULT_COLUMN)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for row, result in zip(rows, results):
            writer.writerow(list(row) + [result])


def load_resume_state(
    input_csv: Path,
    output_csv: Path,
    progress_path: Path,
    row_count: int,
    keyword_column_index: int | None,
) -> tuple[int, list[str]]:
    if not output_csv.exists() or not progress_path.exists():
        return 0, [""] * row_count

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        progress.get("input_csv") != str(input_csv)
        or progress.get("output_csv") != str(output_csv)
        or int(progress.get("total_rows", -1)) != row_count
        or progress.get("keyword_column_index") != keyword_column_index
    ):
        return 0, [""] * row_count
    processed_rows = min(int(progress.get("processed_rows", 0)), row_count)

    with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) - 1 != row_count or not rows or rows[0][-1] != RESULT_COLUMN:
        return 0, [""] * row_count

    results = [(row[-1] if row else "") for row in rows[1:]]
    return processed_rows, results


def save_progress(
    progress_path: Path,
    input_csv: Path,
    output_csv: Path,
    processed_rows: int,
    total_rows: int,
    keyword_column_index: int | None,
) -> None:
    progress = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "processed_rows": processed_rows,
        "total_rows": total_rows,
        "keyword_column_index": keyword_column_index,
    }
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def chunks(start: int, stop: int, size: int) -> Iterable[tuple[int, int]]:
    while start < stop:
        end = min(start + size, stop)
        yield start, end
        start = end


def review_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    reviewer: Callable[[list[str]], list[str] | None],
    batch_size: int = 30,
    keyword_column: str | None = None,
    progress_path: str | Path | None = None,
    max_batches: int | None = None,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    progress_path = Path(progress_path) if progress_path else output_csv.with_suffix(output_csv.suffix + ".progress.json")

    header, rows = read_csv(input_csv)
    column_index = resolve_keyword_column(header, keyword_column)
    processed_rows, results = load_resume_state(input_csv, output_csv, progress_path, len(rows), column_index)

    batches_done = 0
    for start, end in chunks(processed_rows, len(rows), batch_size):
        terms = [keyword_from_row(row, column_index) for row in rows[start:end]]
        request_terms = [term for term in terms if term]
        review_result = reviewer(request_terms) if request_terms else []

        if review_result is None:
            for row_index in range(start, end):
                results[row_index] = MANUAL_REVIEW_MARK
        else:
            non_sensitive = set(item.strip() for item in review_result)
            for offset, term in enumerate(terms):
                if term and term in non_sensitive:
                    results[start + offset] = NON_SENSITIVE_MARK

        processed_rows = end
        write_output(output_csv, header, rows, results)
        save_progress(progress_path, input_csv, output_csv, processed_rows, len(rows), column_index)

        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break

    if not rows:
        write_output(output_csv, header, rows, results)
        save_progress(progress_path, input_csv, output_csv, 0, 0, column_index)
    return output_csv


def build_prompt(terms: list[str]) -> str:
    categories = """
审查目标：按照中国大模型备案/生成式人工智能服务安全评估口径，检查这些拦截关键词中哪些不属于敏感词。

判定原则：敏感词判定必须非常严格。只有词面本身不依赖上下文、一眼就能看出是敏感的，才认为是敏感词，并且不要返回。不要因为联想、最坏情况、隐喻可能性、黑话可能性或少数语境下可能敏感，就把一个词判成敏感词。

凡是不满足“一眼明显敏感”的输入词，都属于不敏感词，必须返回。包括但不限于：一词多义、中性词、历史/政治/学术概念、正常科学语境下不敏感的词、只有在某些上下文中才可能敏感的词、意义不明的数字/编号/手机号样式字符串。

example：下列词单独作为关键词时不属于敏感词，应该返回：三民主义、中共、伪科学、共产党、分裂、假鈔、监听、窃听、国际关系、32663、13366636936。

风险类别只用于判断“词面本身明显敏感”的情况，不能作为联想扩展的理由。包括但不限于：
A.1 含违反社会主义核心价值观的内容：
a) 煽动颠覆国家政权、推翻社会主义制度；
b) 危害国家安全和利益、损害国家形象；
c) 煽动分裂国家、破坏国家统一和社会稳定；
d) 宣扬恐怖主义、极端主义；
e) 宣扬民族仇恨、民族歧视；
f) 宣扬暴力、淫秽色情，或涉及性交易、色情暗示、低俗擦边内容；
g) 传播虚假有害信息；
h) 其他法律、行政法规禁止的内容。

注意：只有明确宣扬、煽动、组织、教唆、实施上述风险，或词面直接呈现色情、暴力、恐怖主义、极端主义、违法犯罪等明显风险时，才作为敏感词不返回。

输出要求：只返回 JSON 字符串数组；数组中只能包含输入词里不属于敏感词的词；不要解释、不要分类、不要新增词。

如果全是不属于敏感词，则返回全部输入词；如果全是敏感词，则返回空列表[]即可。
""".strip()
    return f"{categories}\n\n输入词列表：\n{json.dumps(terms, ensure_ascii=False)}"


def make_model_reviewer(model_name: str | None = None) -> Callable[[list[str]], list[str] | None]:
    from manyllm import ChatSession
    from llm_orchestrator import LLMOrchestrator

    chat = ChatSession()
    orchestrator = LLMOrchestrator(pool=chat.pool, strategy=chat.strategy)

    def reviewer(terms: list[str]) -> list[str] | None:
        if not terms:
            return []
        prompt = build_prompt(terms)
        # print(prompt)
        result = orchestrator.chat(
            [{"role": "user", "content": prompt}],
            system_prompt="你是中国大模型备案安全评估语境下的关键词审查助手。只返回 JSON 数组。",
            temperature=0.0,
            max_tokens=4096,
            thinking=0,
            model_name=model_name,
        )
        if result["status"] != "success":
            raise RuntimeError(result["message"])
        print("大模型返回结果", result["content"])
        try:
            return parse_non_sensitive_terms(result["content"])
        except ValueError:
            print("模型返回了空内容，本批次将标注为 None 并继续。")
            return None

    return reviewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Use project LLMs to find benign terms in a sensitive-keyword CSV.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=30, help="Terms per model request")
    parser.add_argument("--keyword-column", help="Keyword column name or zero-based index. Defaults to the 关键词 column when present.")
    parser.add_argument("--model-name", help="Optional project model name, e.g. glm-4.5-flash")
    parser.add_argument("--max-batches", type=int, help="Optional limit for testing a small run")
    args = parser.parse_args()

    output = review_csv(
        args.input,
        args.output,
        make_model_reviewer(args.model_name),
        batch_size=args.batch_size,
        keyword_column=args.keyword_column,
        max_batches=args.max_batches,
    )
    print(output)


if __name__ == "__main__":
    main()







