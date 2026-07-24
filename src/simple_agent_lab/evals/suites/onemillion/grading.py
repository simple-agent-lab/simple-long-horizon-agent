"""OneMillion-Bench rubric grading — the suite-specific scoring helpers.

This is the ``patch.py`` analog for the OneMillion-Bench suite: the small,
stdlib-only routines the container half needs to grade a model response against
a case's weighted rubrics. The logic is ported verbatim (Chinese judge prompt
included) from the upstream ``omb.grading`` package so the in-environment
``evaluate`` hook reproduces the official rubric scoring without depending on
the vendored ``omb`` checkout (which is not shipped in the wheel).

It imports only the standard library, so it runs inside any eval environment
(in-process or in a bare container) with no copied files. The three upstream
pieces are kept as separate functions:

- ``build_grading_prompt`` — render the judge prompt (with/without human scores).
- ``parse_grading_response`` — robustly extract the judge's per-rubric verdict.
- ``convert_scores`` — turn binary hits into weighted per-rubric scores.

``score_summary`` adds the per-case aggregation (total / max / min / accuracy)
that the host's ``omb`` reporting computes, so one run's ``result.json`` carries
a self-contained verdict.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List

__all__ = [
    "build_grading_prompt",
    "clean_and_parse_json",
    "convert_scores",
    "parse_grading_response",
    "score_summary",
]


# --------------------------------------------------------------------------- #
# Prompt construction (ported from omb.grading.prompts)
# --------------------------------------------------------------------------- #
def build_grading_prompt(
    prompt: str,
    model_response: str,
    rubrics: List[Dict[str, Any]],
    human_scores_dict: Dict[str, int] | None = None,
) -> str:
    """Build the judge prompt for evaluating a model response against rubrics."""

    human_scores_dict = human_scores_dict or {}
    has_human = len(human_scores_dict) > 0

    rubrics_parts = []
    for rubric in rubrics:
        rubric_num = rubric["rubric_number"]
        rubric_detail = rubric["rubric_detail"]
        rubric_weight = rubric["rubric_weight"]

        part = f"""**Rubric {rubric_num}**
rubricDetail: {rubric_detail}
rubricWeight: {rubric_weight:+d}分"""

        if has_human:
            human_key = f"rubric_{rubric_num}_human_score"
            human_score = human_scores_dict.get(human_key, 0)
            human_text = "是" if human_score == rubric_weight else "否"
            part += f"\nhumanScore: {human_text}"

        rubrics_parts.append(part)

    rubrics_str = "\n\n".join(rubrics_parts)

    if has_human:
        return _template_with_human(prompt, model_response, rubrics_str)
    return _template_without_human(prompt, model_response, rubrics_str)


def _template_with_human(prompt: str, model_response: str, rubrics_str: str) -> str:
    return f"""## 角色与核心任务
**角色：** 你是一名公正、精确且严格的AI响应评估裁判。
**核心任务：** 根据详细的评分标准（Rubric），对大型语言模型的回复（modelResponse）进行逐项评估。你需要判断模型回复是否符合评分标准中的具体描述，并给出评估结果。
**评估原则：**
1. **寻找直接证据：** 评估必须严格依据模型回复中**实际存在**的文本证据。不能进行主观猜测或过度解读。只有明确指出的内容才算数。
2. **二元判断（是/否）：** 每一个 Rubric 项的评估结果只有两种：
   - **命中 (是)**：模型回复中确实包含或命中了了rubric描述的内容或特征。
   - **未命中 (否)**：模型回复中没有包含或没有命中rubric描述的内容或特征。
   *注意：这一逻辑通用于正分项（得分点）和负分项（扣分点）。只要rubric里的描述发生了，就是"命中/是"。*

## 评分步骤
请保持冷静和专注，严格遵循以下步骤：
**步骤一：理解上下文**
仔细阅读用户问题（prompt）、模型回复（modelResponse）以及评分标准（rubric）。
**步骤二：判断是否命中**
对照评分标准（rubric）的描述，检查模型回复：
- 如果回复中**出现**了rubric描述的情况（无论是好的行为还是坏的错误），状态为 **"命中"**，结论输出 **"是"**。
- 如果回复中**未出现**rubric描述的情况，状态为 **"未命中"**，结论输出 **"否"**。
**步骤三：自我反思与格式化**
- 检查证据是否充分支持你的"是/否"判断。
- 比较你的判断（status）与参考的人工评分（humanScore，如果存在）：
  - 如果一致，consistency字段填"Consistent"
  - 如果不一致，consistency字段填"Inconsistent"
- 严格按照JSON格式输出。

## 输出格式
对每条Rubric，输出一个JSON对象，包含以下字段：

```json
[
  {{
    "rubric_id": 1,
    "status": "是",
    "justification": "模型回复完整说明了问题X，符合评分要求",
    "consistency": "Consistent"
  }},
  {{
    "rubric_id": 2,
    "status": "否",
    "justification": "模型回复未提及关键点Y",
    "consistency": "Inconsistent"
  }}
]
```

**字段说明**：
- `rubric_id`：Rubric编号
- `status`：**"是"** 或 **"否"**
- `justification`：简要的中文评估依据（1-2句话）
- `consistency`：**"Consistent"** 或 **"Inconsistent"** (仅当提供humanScore时)

---

## 输入信息

### 用户问题（prompt）
{prompt}

---

### AI回复（modelResponse）
{model_response}

---

### 评分项（Rubrics）
{rubrics_str}

---

请逐条评估所有Rubric，输出完整JSON数组。"""


def _template_without_human(prompt: str, model_response: str, rubrics_str: str) -> str:
    return f"""## 角色与核心任务

**角色：** 你是一名公正、精确且严格的AI响应评估裁判。

**核心任务：** 根据详细的评分标准（Rubric），对大型语言模型的回复（modelResponse）进行逐项评估。你需要判断模型回复是否符合评分标准中的具体描述。

**评估原则：**

1. **寻找直接证据：** 评估必须严格依据模型回复中**实际存在**的文本证据。不能进行主观猜测或过度解读。只有明确指出的内容才算数。

2. **二元判断（是/否）：** 每一个 Rubric 项的评估结果只有两种：
   - **命中 (是)**：模型回复中确实包含或命中了rubric描述的内容或特征。
   - **未命中 (否)**：模型回复中没有包含或没有命中rubric描述的内容或特征。

   *注意：这一逻辑通用于正分项（得分点）和负分项（扣分点）。只要rubric里的描述发生了，就是"命中/是"。*

---

## 评分步骤

请保持冷静和专注，严格遵循以下步骤：

**步骤一：理解上下文**
仔细阅读用户问题（prompt）、模型回复（modelResponse）、评分标准（rubric）。

**步骤二：判断是否命中**
对照评分标准（rubric）的描述，检查模型回复：
- 如果回复中**出现**了rubric描述的情况（无论是好的行为还是坏的错误），状态为 **"命中"**，结论输出 **"是"**。
- 如果回复中**未出现**rubric描述的情况，状态为 **"未命中"**，结论输出 **"否"**。

**步骤三：自我反思与格式化**
- 检查证据是否充分支持你的"是/否"判断。
- 严格按照JSON格式输出。

---

## 输出格式

对每条Rubric，输出一个JSON对象，包含以下字段：

```json
[
  {{
    "rubric_id": 1,
    "status": "是",
    "justification": "模型回复完整说明了问题X，符合评分要求"
  }},
  {{
    "rubric_id": 2,
    "status": "否",
    "justification": "模型回复未提及关键点Y"
  }}
]
```

**字段说明**：
- `rubric_id`：Rubric编号
- `status`：**"是"** 或 **"否"**
- `justification`：简要的中文评估依据（1-2句话）

---

## 输入信息

### 用户问题（prompt）
{prompt}

---

### AI回复（modelResponse）
{model_response}

---

### 评分项（Rubrics）
{rubrics_str}

---

请逐条评估所有Rubric，输出完整JSON数组。"""


# --------------------------------------------------------------------------- #
# Response parsing (ported from omb.grading.parser)
# --------------------------------------------------------------------------- #
def _warn(message: str) -> None:
    """Stderr warning — replaces omb's rich ``print_warning`` (stdlib-only here)."""

    print(f"[onemillion grading] {message}", file=sys.stderr)


def clean_and_parse_json(json_str: str) -> Any:
    """Attempt to clean and parse a JSON string (smart quotes, trailing commas)."""

    if not json_str:
        raise ValueError("Empty JSON string")

    json_str = json_str.strip()
    fixed_commas = re.sub(r",\s*([}\]])", r"\1", json_str)
    fixed_quotes = json_str.replace("“", '"').replace("”", '"')
    candidates = (
        json_str,
        fixed_commas,
        fixed_quotes,
        re.sub(r",\s*([}\]])", r"\1", fixed_quotes),
    )
    if json_str.startswith("[") and not json_str.endswith("]"):
        candidates += (json_str + "]",)
    for candidate in dict.fromkeys(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON")


def _json_array_candidates(response_text: str) -> list[tuple[str, int]]:
    """Return fenced, balanced, then broad JSON-array candidates."""

    candidates = [
        (match.group(1), pattern)
        for pattern, regex in enumerate(
            (r"```json\s*(\[[\s\S]*\])\s*```", r"```\s*(\[[\s\S]*\])\s*```"),
            1,
        )
        if (match := re.search(regex, response_text, re.DOTALL))
    ]
    for start in (match.start() for match in re.finditer(r"\[", response_text)):
        depth = 0
        in_string = False
        escaped = False
        for end, char in enumerate(response_text[start:], start):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                depth += (char == "[") - (char == "]")
                if depth == 0:
                    candidates.append((response_text[start : end + 1], 3))
                    break
    first, last = response_text.find("["), response_text.rfind("]")
    if 0 <= first < last:
        candidates.append((response_text[first : last + 1], 4))
    return candidates


def _rubric_key(value: Any) -> Any:
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def parse_grading_response(
    response_text: str, rubrics: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """Parse the judge response into a per-rubric verdict dict."""

    results: Dict[Any, Dict[str, Any]] = {}
    json_str: str | None = None
    data_list: Any = None
    matched_pattern: int | None = None
    for candidate, pattern in _json_array_candidates(response_text):
        try:
            data_list = clean_and_parse_json(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        json_str, matched_pattern = candidate, pattern
        break

    if json_str is None:
        _warn(f"no JSON array found; first 500 chars: {response_text[:500]}")
    else:
        try:
            for item in data_list:
                rubric_id = _rubric_key(item.get("rubric_id"))
                status = item.get("status", "").strip()
                if rubric_id is not None and rubric_id != "":
                    results[rubric_id] = {
                        "status": status,
                        "binary_score": int(
                            status
                            in {"是", "Yes", "yes", "Y", "YES", "true", "True", "命中"}
                        ),
                        "justification": item.get("justification", "").strip(),
                        "consistency": item.get("consistency", "").strip(),
                    }
        except Exception as exc:  # noqa: BLE001 - keep grading resilient
            _warn(f"JSON parse error: {str(exc)[:100]}")
            _warn(f"matched pattern: {matched_pattern}, json length: {len(json_str)}")

    for rubric in rubrics:
        lookup_key = _rubric_key(rubric["rubric_number"])
        if lookup_key not in results:
            results[lookup_key] = {
                "status": "否",
                "binary_score": "NA",
                "justification": "解析失败",
                "consistency": "",
            }

    return results


def convert_scores(
    raw_results: Dict[int, Dict[str, Any]], rubrics: List[Dict[str, Any]]
) -> Dict[Any, Any]:
    """Convert binary hit/miss verdicts into weighted per-rubric scores."""

    final_scores: Dict[Any, Any] = {}

    for rubric in rubrics:
        rubric_num = rubric["rubric_number"]
        weight = rubric["rubric_weight"]

        lookup_key = _rubric_key(rubric_num)
        raw_result = raw_results.get(lookup_key, {"binary_score": 0})
        binary_score = raw_result["binary_score"]

        if binary_score == "NA":
            final_scores[rubric_num] = "NA"
            continue

        final_scores[rubric_num] = weight if binary_score == 1 else 0

    return final_scores


# --------------------------------------------------------------------------- #
# Per-case aggregation (mirrors omb.orchestrator scoring math)
# --------------------------------------------------------------------------- #
def score_summary(
    final_scores: Dict[Any, Any], rubrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate per-rubric scores into one case verdict.

    Mirrors the per-task math in ``omb``'s reporting:

    - ``total_score`` — sum of non-``NA`` weighted scores.
    - ``max_score`` / ``min_score`` — sum of positive / negative rubric weights.
    - ``accuracy`` — ``total / max`` when there are positive points; for a
      pure-penalty case (``max == 0`` and ``min < 0``) it is ``1 - total/min``.
    """

    total_score = sum(s for s in final_scores.values() if s != "NA")
    max_score = sum(r["rubric_weight"] for r in rubrics if r["rubric_weight"] > 0)
    min_score = sum(r["rubric_weight"] for r in rubrics if r["rubric_weight"] < 0)

    if max_score > 0:
        accuracy = total_score / max_score
    elif max_score == 0 and min_score < 0:
        accuracy = 1 - (total_score / min_score)
    else:
        accuracy = 0.0

    return {
        "total_score": total_score,
        "max_score": max_score,
        "min_score": min_score,
        "accuracy": accuracy,
    }
