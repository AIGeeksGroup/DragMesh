#!/usr/bin/env python3
"""用 OpenAI-compatible API 批量评估 25 个交互 case 的关节类型。

输入 JSON 或 CSV 需要包含字段：
- id 或 case_id
- category
- interaction
- gt_joint_type: revolute / prismatic / static

示例：
python scripts/run_llm_joint_type_eval.py \
  --input results/gap25_joint_type_cases.json \
  --output_dir results/llm_joint_type_eval \
  --model gpt-4o
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

VALID_JOINT_TYPES = {"revolute", "prismatic", "static"}

SYSTEM_PROMPT = (
    "You are an expert in 3D object kinematics. Classify the joint type into "
    "exactly one of the following: revolute, prismatic, or static. "
    "Output ONLY ONE WORD. No punctuation, no explanation."
)

USER_PROMPT_TEMPLATE = (
    "Object Category: {category}. Interaction: {interaction}. "
    "What is the joint type? Output exactly one word."
)


@dataclass(frozen=True)
class TestCase:
    """单个关节类型测试样本。"""

    case_id: str
    category: str
    interaction: str
    gt_joint_type: str


@dataclass
class PredictionRow:
    """单个样本的模型预测与评估结果。"""

    case_id: str
    category: str
    interaction: str
    gt_joint_type: str
    pred_joint_type: str
    raw_response: str
    correct: bool
    error: str


def normalize_joint_type(value: object) -> str:
    """把 API 返回或 GT 字段规范化到三个合法类别之一。"""

    text = str(value or "").strip().lower()
    text = text.replace(".", "").replace(",", "")
    if text in VALID_JOINT_TYPES:
        return text
    return "error"


def read_json_cases(path: Path) -> List[TestCase]:
    """读取 JSON 数据；支持 list 或 {"cases": [...]} 两种格式。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"JSON 输入应为 list 或包含 cases 的 dict: {path}")
    return [row_to_case(row) for row in rows]


def read_csv_cases(path: Path) -> List[TestCase]:
    """读取 CSV 数据。"""

    with path.open(newline="", encoding="utf-8") as f:
        return [row_to_case(row) for row in csv.DictReader(f)]


def row_to_case(row: Dict[str, object]) -> TestCase:
    """把原始行转换为强类型 TestCase，并做必要字段检查。"""

    case_id = str(row.get("id") or row.get("case_id") or "").strip()
    category = str(row.get("category") or "").strip()
    interaction = str(row.get("interaction") or "").strip()
    gt_joint_type = normalize_joint_type(row.get("gt_joint_type"))

    missing = []
    if not case_id:
        missing.append("id/case_id")
    if not category:
        missing.append("category")
    if not interaction:
        missing.append("interaction")
    if gt_joint_type == "error":
        missing.append("gt_joint_type")
    if missing:
        raise ValueError(f"case 字段缺失或非法: {missing}; row={row}")

    return TestCase(
        case_id=case_id,
        category=category,
        interaction=interaction,
        gt_joint_type=gt_joint_type,
    )


def load_cases(path: Path) -> List[TestCase]:
    """根据扩展名自动读取 JSON 或 CSV。"""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json_cases(path)
    if suffix == ".csv":
        return read_csv_cases(path)
    raise ValueError(f"只支持 .json 或 .csv 输入: {path}")


def call_llm_joint_type(
    *,
    client: object,
    model: str,
    case: TestCase,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    """调用 OpenAI-compatible Chat Completions API，并带指数退避重试。"""

    user_prompt = USER_PROMPT_TEMPLATE.format(
        category=case.category,
        interaction=case.interaction,
    )

    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            # openai>=1.0.0 的官方 SDK 风格：client.chat.completions.create(...)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=8,
            )
            content = response.choices[0].message.content
            return str(content or "")
        except Exception as exc:  # noqa: BLE001 - API 兼容端可能抛出不同异常类型
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_s = retry_sleep * (2**attempt)
            print(f"[retry] {case.case_id}: attempt={attempt + 1}, sleep={sleep_s:.1f}s, error={exc}")
            time.sleep(sleep_s)

    raise RuntimeError(f"API 调用失败: case_id={case.case_id}, error={last_error}")


def predict_case(
    *,
    client: object,
    model: str,
    case: TestCase,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> PredictionRow:
    """预测单个 case，并做 deterministic parsing。"""

    try:
        raw_response = call_llm_joint_type(
            client=client,
            model=model,
            case=case,
            temperature=temperature,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        pred = normalize_joint_type(raw_response)
        error = "" if pred != "error" else "invalid_model_output"
    except Exception as exc:  # noqa: BLE001
        raw_response = ""
        pred = "error"
        error = str(exc)

    return PredictionRow(
        case_id=case.case_id,
        category=case.category,
        interaction=case.interaction,
        gt_joint_type=case.gt_joint_type,
        pred_joint_type=pred,
        raw_response=raw_response,
        correct=pred == case.gt_joint_type,
        error=error,
    )


def write_predictions_csv(path: Path, rows: Iterable[PredictionRow]) -> None:
    """保存每个 case 的完整预测结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "category",
        "interaction",
        "gt_joint_type",
        "pred_joint_type",
        "raw_response",
        "correct",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_failure_cases(path: Path, rows: List[PredictionRow]) -> None:
    """保存错误样本，便于后续 error analysis。"""

    failures = [
        {
            "id": row.case_id,
            "category": row.category,
            "interaction": row.interaction,
            "gt_joint_type": row.gt_joint_type,
            "pred_joint_type": row.pred_joint_type,
            "raw_response": row.raw_response,
            "error": row.error,
        }
        for row in rows
        if not row.correct
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, rows: List[PredictionRow], model: str) -> None:
    """保存总体 accuracy 和逐类统计。"""

    total = len(rows)
    correct = sum(row.correct for row in rows)
    errors = sum(row.pred_joint_type == "error" for row in rows)

    per_category: Dict[str, Dict[str, object]] = {}
    for category in sorted({row.category for row in rows}):
        subset = [row for row in rows if row.category == category]
        n = len(subset)
        c = sum(row.correct for row in subset)
        per_category[category] = {
            "total": n,
            "correct": c,
            "accuracy": c / n if n else 0.0,
        }

    summary = {
        "model": model,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total if total else 0.0,
        "per_category": per_category,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_client(api_key: Optional[str], base_url: Optional[str]) -> object:
    """构造 OpenAI SDK client；base_url 可指向兼容 OpenAI 协议的内部 API。"""

    from openai import OpenAI

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise EnvironmentError("缺少 API key：请设置 OPENAI_API_KEY 或传入 --api_key")

    if base_url:
        return OpenAI(api_key=resolved_key, base_url=base_url)
    return OpenAI(api_key=resolved_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="测试集 JSON/CSV，包含 id/category/interaction/gt_joint_type")
    parser.add_argument("--output_dir", type=Path, default=Path("results/llm_joint_type_eval"))
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, default=None, help="默认读取 OPENAI_API_KEY")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible API base URL，可选")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="调试用：只跑前 N 个 case")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]

    client = build_client(api_key=args.api_key, base_url=args.base_url)
    rows: List[PredictionRow] = []

    for idx, case in enumerate(cases, start=1):
        row = predict_case(
            client=client,
            model=args.model,
            case=case,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        rows.append(row)
        status = "OK" if row.correct else "FAIL"
        print(
            f"[{idx:02d}/{len(cases):02d}] {case.case_id}: "
            f"gt={row.gt_joint_type}, pred={row.pred_joint_type}, {status}"
        )

    predictions_csv = args.output_dir / "predictions.csv"
    failures_json = args.output_dir / "failure_cases.json"
    summary_json = args.output_dir / "summary.json"
    write_predictions_csv(predictions_csv, rows)
    write_failure_cases(failures_json, rows)
    write_summary(summary_json, rows, args.model)

    correct = sum(row.correct for row in rows)
    total = len(rows)
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.3f}")
    print(f"Wrote {predictions_csv}")
    print(f"Wrote {failures_json}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
