import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.review_sensitive_keywords import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    build_prompt,
    parse_non_sensitive_terms,
    review_csv,
)


RESULT_COLUMN = "\u5224\u522b\u7ed3\u679c"
MARK = "\u4e0d\u5c5e\u4e8e\u654f\u611f\u8bcd"


class SensitiveKeywordReviewTest(unittest.TestCase):
    def test_default_input_uses_latest_attachment4_csv(self):
        self.assertEqual(DEFAULT_INPUT, Path("dataset") / "附件4-拦截关键词0727.csv")

    def test_default_output_filename_is_ascii_safe(self):
        str(DEFAULT_OUTPUT).encode("ascii")
        self.assertEqual(DEFAULT_OUTPUT.name, "sensitive_keyword_model_review.csv")

    def test_build_prompt_uses_strict_obvious_sensitive_threshold(self):
        prompt = build_prompt(["\u6843\u8272\u4ea4\u6613", "\u82f9\u679c"])

        self.assertIn("\u4e2d\u56fd\u5927\u6a21\u578b\u5907\u6848", prompt)
        self.assertIn("A.1", prompt)
        self.assertIn("\u793e\u4f1a\u4e3b\u4e49\u6838\u5fc3\u4ef7\u503c\u89c2", prompt)
        self.assertIn("\u5ba3\u626c\u66b4\u529b\u3001\u6deb\u79fd\u8272\u60c5", prompt)
        self.assertIn("\u6843\u8272\u4ea4\u6613", prompt)
        self.assertIn("\u4e00\u773c\u5c31\u80fd\u770b\u51fa\u662f\u654f\u611f", prompt)
        self.assertIn("\u4e00\u8bcd\u591a\u4e49", prompt)
        self.assertIn("\u6b63\u5e38\u79d1\u5b66\u8bed\u5883", prompt)
        self.assertIn("\u610f\u4e49\u4e0d\u660e\u7684\u6570\u5b57", prompt)
        for term in ["三民主义", "中共", "伪科学", "共产党", "分裂", "假鈔", "监听", "窃听", "国际关系", "32663", "13366636936"]:
            self.assertIn(term, prompt)

    def test_review_csv_defaults_to_keyword_header_column(self):
        calls = []

        def reviewer(batch):
            calls.append(list(batch))
            return []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "keywords.csv"
            output_csv = tmp_path / "keywords_reviewed.csv"

            with input_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["\u5e8f\u53f7", "\u7c7b\u522b", "\u5c0f\u7c7b", "\u5173\u952e\u8bcd"])
                writer.writerows(
                    [
                        ["1301", "A.1", "f", "\u6843\u8272\u4ea4\u6613"],
                        ["1302", "A.1", "f", "\u82f9\u679c"],
                    ]
                )

            review_csv(input_csv, output_csv, reviewer, batch_size=100)

            self.assertEqual(calls, [["\u6843\u8272\u4ea4\u6613", "\u82f9\u679c"]])
    def test_parse_non_sensitive_terms_rejects_none_response(self):
        with self.assertRaisesRegex(ValueError, "empty model response"):
            parse_non_sensitive_terms(None)
    def test_parse_non_sensitive_terms_accepts_json_array_and_plain_lines(self):
        self.assertEqual(parse_non_sensitive_terms('["apple", "banana"]'), ["apple", "banana"])
        self.assertEqual(parse_non_sensitive_terms("- apple\n- banana"), ["apple", "banana"])

    def test_review_csv_marks_entire_batch_none_when_reviewer_returns_none(self):
        calls = []

        def reviewer(batch):
            calls.append(list(batch))
            return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "keywords.csv"
            output_csv = tmp_path / "keywords_reviewed.csv"
            progress = tmp_path / "keywords_reviewed.progress.json"

            with input_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["keyword"])
                writer.writerows([["one"], ["two"], ["three"]])

            review_csv(input_csv, output_csv, reviewer, batch_size=2, progress_path=progress)

            with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(calls, [["one", "two"], ["three"]])
            self.assertEqual(rows[1], ["one", "None"])
            self.assertEqual(rows[2], ["two", "None"])
            self.assertEqual(rows[3], ["three", "None"])
            self.assertEqual(json.loads(progress.read_text(encoding="utf-8"))["processed_rows"], 3)

    def test_review_csv_ignores_stale_resume_state_for_different_input(self):
        calls = []

        def reviewer(batch):
            calls.append(list(batch))
            return list(batch)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "new_keywords.csv"
            output_csv = tmp_path / "keywords_reviewed.csv"
            progress = tmp_path / "keywords_reviewed.progress.json"

            with input_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["关键词"])
                writer.writerows([["三民主义"], ["32663"]])

            output_csv.write_bytes("序号,关键词,判别结果\r\n1,旧词,\r\n".encode("gbk"))
            progress.write_text(
                json.dumps(
                    {
                        "input_csv": str(tmp_path / "old_keywords.csv"),
                        "output_csv": str(output_csv),
                        "processed_rows": 1,
                        "total_rows": 1,
                        "keyword_column_index": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            review_csv(input_csv, output_csv, reviewer, batch_size=100, progress_path=progress)

            self.assertEqual(calls, [["三民主义", "32663"]])
            self.assertEqual(json.loads(progress.read_text(encoding="utf-8"))["input_csv"], str(input_csv))

    def test_review_csv_batches_and_resumes_from_progress(self):
        calls = []

        def reviewer(batch):
            calls.append(list(batch))
            return [batch[0]]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "keywords.csv"
            output_csv = tmp_path / "keywords_reviewed.csv"
            progress = tmp_path / "keywords_reviewed.progress.json"

            with input_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["keyword", "source"])
                writer.writerows(
                    [
                        ["apple", "a"],
                        ["badword", "b"],
                        ["banana", "c"],
                        ["blocked", "d"],
                        ["cat", "e"],
                    ]
                )

            review_csv(
                input_csv,
                output_csv,
                reviewer,
                batch_size=2,
                progress_path=progress,
                max_batches=1,
            )

            self.assertEqual(calls, [["apple", "badword"]])
            self.assertEqual(json.loads(progress.read_text(encoding="utf-8"))["processed_rows"], 2)

            review_csv(
                input_csv,
                output_csv,
                reviewer,
                batch_size=2,
                progress_path=progress,
            )

            with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(calls, [["apple", "badword"], ["banana", "blocked"], ["cat"]])
            self.assertEqual(rows[0], ["keyword", "source", RESULT_COLUMN])
            self.assertEqual(rows[1], ["apple", "a", MARK])
            self.assertEqual(rows[2], ["badword", "b", ""])
            self.assertEqual(rows[3], ["banana", "c", MARK])
            self.assertEqual(rows[4], ["blocked", "d", ""])
            self.assertEqual(rows[5], ["cat", "e", MARK])






