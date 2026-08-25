from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from evaluate import judge as official_judge
from src.client import call_llm
from src.memory.few_shot import retrieve_examples
from src.memory.knowledge import retrieve_knowledge
from src.tools.tabtools_mimic3 import (
    data_filter,
    date_calculator,
    db_loader,
    get_value,
    sql_interpreter,
)


TOOLS_MAP = {
    "LoadDB": db_loader,
    "FilterDB": data_filter,
    "GetValue": get_value,
    "SQLInterpreter": sql_interpreter,
    "Calendar": date_calculator,
}


SYSTEM_PROMPT_COT = """You are an EHR clinical reasoning agent solving MIMIC-III questions.
This is the EHRAgent ablation without interactive Python coding.
Do not generate or execute Python code. Use natural-language planning and one external tool action at a time.

Available tools:
- LoadDB[table]
- FilterDB[database_variable, condition]
- GetValue[database_variable, column or column, operation]
- SQLInterpreter[SQL query]
- Calendar[duration]

On the first turn, state a short numbered Plan and execute its first action.
On every action turn, use exactly this format:
Thought: <reason for this step>
Action: ToolName[arguments]
Variable: <name used by later steps>

Use literal values from observations or {variable_name} placeholders in later actions.
You must execute at least one tool action before answering. Never guess or emit placeholders.
When the tool evidence is sufficient, output only:
Thought: <brief evidence-based conclusion>
Final Answer: <exact answer value only>
TERMINATE
"""


_COT_SEED_MEMORY = [
    {
        "question": "What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?",
        "knowledge": "Find the diagnosis code, matching admissions, and their costs after the reference date.",
        "code": """Plan: resolve the date, diagnosis code, admission IDs, and maximum cost.
Action: Calendar[-1 year]
Variable: ref_date
Action: SQLInterpreter[SELECT MAX(cost) FROM cost WHERE chargetime >= '{ref_date}' AND hadm_id IN (SELECT hadm_id FROM diagnoses_icd WHERE icd9_code IN (SELECT icd9_code FROM d_icd_diagnoses WHERE short_title = 'comp-oth vasc dev/graft'))]
Variable: max_cost""",
    },
    {
        "question": "had any tpn w/lipids been given to patient 2238 in their last hospital visit?",
        "knowledge": "Resolve the last admission, the item identifier, and matching input events.",
        "code": """Plan: find the last admission and count matching input events.
Action: SQLInterpreter[SELECT COUNT(*) FROM inputevents_cv WHERE hadm_id = (SELECT hadm_id FROM admissions WHERE subject_id = 2238 ORDER BY dischtime DESC LIMIT 1) AND itemid IN (SELECT itemid FROM d_items WHERE label = 'tpn w/lipids')]
Variable: event_count""",
    },
    {
        "question": "what was the name of the procedure that was given two or more times to patient 58730?",
        "knowledge": "Count procedure codes for the patient and map qualifying codes to procedure names.",
        "code": """Plan: aggregate procedure codes and retrieve their short titles.
Action: SQLInterpreter[SELECT short_title FROM d_icd_procedures WHERE icd9_code IN (SELECT icd9_code FROM procedures_icd WHERE hadm_id IN (SELECT hadm_id FROM admissions WHERE subject_id = 58730) GROUP BY icd9_code HAVING COUNT(*) >= 2)]
Variable: procedure_names""",
    },
    {
        "question": "calculate the length of stay of the first stay of patient 27392 in the icu.",
        "knowledge": "Use the first ICU stay timestamps and compute the rounded day difference.",
        "code": """Plan: select the first ICU stay and compute its duration in days.
Action: SQLInterpreter[SELECT CAST(julianday(outtime) - julianday(intime) + CASE WHEN ((julianday(outtime) - julianday(intime)) - CAST(julianday(outtime) - julianday(intime) AS INTEGER)) > 0.5 THEN 1 ELSE 0 END AS INTEGER) FROM icustays WHERE subject_id = 27392 ORDER BY intime ASC LIMIT 1]
Variable: stay_days""",
    },
]


@dataclass(frozen=True, slots=True)
class ToolStepResult:
    value: object | None
    observation: str
    error: str | None = None


class _LLMMessage(Protocol):
    content: str | None


class _LLMChoice(Protocol):
    message: _LLMMessage


class _LLMResponse(Protocol):
    choices: list[_LLMChoice]


def build_cot_seed_memory() -> list[dict[str, str]]:
    return [item.copy() for item in _COT_SEED_MEMORY]


def parse_action(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"Action:\s*([A-Za-z_]\w*)\s*\[([^\n]*)\]", text)
    if match is None:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def parse_variable(text: str, turn: int) -> str:
    match = re.search(r"Variable:\s*([A-Za-z_]\w*)", text)
    return match.group(1).strip() if match else f"var_{turn}"


def parse_final_answer(text: str) -> str | None:
    match = re.search(
        r"Final Answer:\s*(.+?)(?:\n|TERMINATE|$)", text, re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def _strip_outer_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1]
    return stripped


def _substitute_variables(text: str, variables: dict[str, object]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = variables.get(name)
        if value is None or isinstance(value, pd.DataFrame):
            return match.group(0)
        return str(value)

    return re.sub(r"\{([A-Za-z_]\w*)\}", replace, text)


def _format_observation(value: object) -> str:
    if isinstance(value, pd.DataFrame):
        columns = ", ".join(str(column) for column in value.columns)
        return f"table with {len(value)} rows; columns: {columns}"
    rendered = str(value)
    return rendered if len(rendered) <= 4000 else rendered[:4000] + "..."


def execute_tool_step(
    tool_name: str,
    args_str: str,
    var_store: dict[str, object],
) -> ToolStepResult:
    if tool_name not in TOOLS_MAP:
        available = ", ".join(sorted(TOOLS_MAP))
        message = f"Tool '{tool_name}' is not recognized. Available tools: {available}."
        return ToolStepResult(value=None, observation=f"Error: {message}", error=message)

    try:
        if tool_name == "LoadDB":
            value = db_loader(_strip_outer_quotes(args_str))
        elif tool_name == "Calendar":
            substituted = _substitute_variables(args_str, var_store)
            value = date_calculator(_strip_outer_quotes(substituted))
        elif tool_name == "SQLInterpreter":
            substituted = _substitute_variables(args_str, var_store)
            value = sql_interpreter(_strip_outer_quotes(substituted))
        elif tool_name in {"FilterDB", "GetValue"}:
            variable_name, separator, argument = args_str.partition(",")
            if not separator:
                message = f"{tool_name} requires a database variable and an argument."
                return ToolStepResult(
                    value=None, observation=f"Error: {message}", error=message
                )
            data = var_store.get(variable_name.strip())
            if not isinstance(data, pd.DataFrame):
                message = f"Variable '{variable_name.strip()}' is not a loaded table."
                return ToolStepResult(
                    value=None, observation=f"Error: {message}", error=message
                )
            substituted = _substitute_variables(argument, var_store)
            clean_argument = _strip_outer_quotes(substituted)
            value = (
                data_filter(data, clean_argument)
                if tool_name == "FilterDB"
                else get_value(data, clean_argument)
            )
        else:
            message = f"No executor is defined for {tool_name}."
            return ToolStepResult(
                value=None, observation=f"Error: {message}", error=message
            )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return ToolStepResult(value=None, observation=f"Error: {message}", error=message)

    return ToolStepResult(value=value, observation=_format_observation(value))


def _normalize_atom(value: str) -> str:
    normalized = " ".join(_strip_outer_quotes(value).strip().casefold().split())
    aliases = {
        "true": "1",
        "yes": "1",
        "false": "0",
        "no": "0",
        "none": "0",
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return normalized
    return format(number.normalize(), "f")


def strict_judge(prediction: str, answer: str) -> bool:
    pred_parts = [_normalize_atom(part) for part in prediction.split(",")]
    answer_parts = [_normalize_atom(part) for part in answer.split(",")]
    return sorted(pred_parts) == sorted(answer_parts)


def load_items(path: str) -> list[dict[str, object]]:
    data_path = Path(path)
    text = data_path.read_text(encoding="utf-8").strip()
    if data_path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        raise TypeError("Evaluation data must be a list of question records.")
    return loaded


def select_eval_items(
    data_path: str,
    num_questions: int,
    seed: int,
) -> list[dict[str, object]]:
    items = load_items(data_path)
    rng = random.Random(seed)
    rng.shuffle(items)
    count = len(items) if num_questions == -1 else min(num_questions, len(items))
    return items[:count]


def run_question_wo_interactive_coding(
    question: str,
    model_name: str,
    memory: list[dict[str, str]] | None = None,
    max_turns: int = 20,
    knowledge: str = "",
) -> dict[str, object]:
    resolved_knowledge = knowledge or retrieve_knowledge(question, model_name)
    active_memory = memory if memory is not None else build_cot_seed_memory()
    examples = retrieve_examples(question, active_memory)
    user_prompt = f"""Retrieved examples:
{examples}
(END OF EXAMPLES)

Domain knowledge:
{resolved_knowledge}

Question: {question}
Solve the question using the required natural-language plan and step-by-step tools.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_prompt},
    ]
    variables: dict[str, object] = {}
    trace: list[dict[str, object]] = []
    last_answer: str | None = None
    action_count = 0
    error_count = 0
    last_tool_succeeded = False

    for turn in range(max_turns):
        response = cast(
            _LLMResponse,
            cast(
                object,
                call_llm(
                    model_name,
                    messages,
                    label=f"turn {turn} [w/o interactive coding]",
                ),
            ),
        )
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        final_answer = parse_final_answer(content)

        if final_answer is not None and action_count == 0:
            feedback = "A final answer is not accepted before at least one tool action. Execute the first planned tool step."
            trace.append(
                {"turn": turn, "assistant": content, "observation": feedback}
            )
            messages.append({"role": "user", "content": feedback})
            continue

        if final_answer is not None:
            last_answer = final_answer
            terminated = "TERMINATE" in content
            completed = terminated and last_tool_succeeded
            trace.append({"turn": turn, "assistant": content})
            if terminated:
                return {
                    "knowledge": resolved_knowledge,
                    "answer": last_answer,
                    "terminated": True,
                    "completed": completed,
                    "turns": turn + 1,
                    "actions": action_count,
                    "errors": error_count,
                    "trace": trace,
                }
            messages.append(
                {
                    "role": "user",
                    "content": "When finished, repeat the exact Final Answer and append TERMINATE.",
                }
            )
            continue

        tool_name, args_str = parse_action(content)
        if tool_name is None or args_str is None:
            feedback = "Output one Action: ToolName[arguments] with Variable, or a tool-supported Final Answer followed by TERMINATE."
            trace.append(
                {"turn": turn, "assistant": content, "observation": feedback}
            )
            messages.append({"role": "user", "content": feedback})
            continue

        result = execute_tool_step(tool_name, args_str, variables)
        action_count += 1
        last_tool_succeeded = result.error is None
        if result.error is not None:
            error_count += 1
        else:
            variables[parse_variable(content, turn)] = result.value
        observation = f"Observation: {result.observation}"
        trace.append(
            {
                "turn": turn,
                "assistant": content,
                "tool": tool_name,
                "arguments": args_str,
                "observation": result.observation,
                "error": result.error,
            }
        )
        messages.append({"role": "user", "content": observation})

    return {
        "knowledge": resolved_knowledge,
        "answer": last_answer or "No answer produced",
        "terminated": False,
        "completed": False,
        "turns": max_turns,
        "actions": action_count,
        "errors": error_count,
        "trace": trace,
    }


def _gold_string(raw_answer: object) -> str:
    if isinstance(raw_answer, list):
        return ", ".join(str(value) for value in raw_answer)
    return str(raw_answer)


def evaluate_model_wo_interactive_coding(
    model_name: str,
    eval_set: list[dict[str, object]],
    seed: int,
    output_dir: str,
    max_turns: int,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / f"results_{model_name}_{seed}.jsonl"
    summary_path = output_path / f"summary_{model_name}_{seed}.json"
    results_path.write_text("", encoding="utf-8")
    memory = build_cot_seed_memory()
    strict_correct_count = 0
    official_correct_count = 0
    completed_count = 0
    start_time = time.time()

    for index, item in enumerate(eval_set):
        question = str(item["template"])
        gold = _gold_string(item["answer"])
        question_start = time.time()
        outcome = run_question_wo_interactive_coding(
            question,
            model_name,
            memory=memory,
            max_turns=max_turns,
        )
        prediction = str(outcome["answer"])
        strict_correct = strict_judge(prediction, gold)
        repository_correct = bool(official_judge(prediction, gold))
        completed = bool(outcome["completed"])
        strict_correct_count += int(strict_correct)
        official_correct_count += int(repository_correct)
        completed_count += int(completed)

        if strict_correct:
            trace = outcome["trace"]
            if not isinstance(trace, list):
                raise TypeError("Question trace must be a list.")
            plan_trace = "\n".join(
                str(step.get("assistant", ""))
                for step in trace
                if isinstance(step, dict)
            )
            memory.append(
                {
                    "question": question,
                    "knowledge": str(outcome["knowledge"]),
                    "code": plan_trace,
                }
            )

        record = {
            "i": index,
            "id": item["id"],
            "ablation": "wo_interactive_coding_paper_aligned",
            "model": model_name,
            "question": question,
            "gold": gold,
            "pred": prediction,
            "correct": strict_correct,
            "official_correct": repository_correct,
            "terminated": outcome["terminated"],
            "completed": completed,
            "turns": outcome["turns"],
            "actions": outcome["actions"],
            "errors": outcome["errors"],
            "trace": outcome["trace"],
            "elapsed_sec": round(time.time() - question_start, 2),
        }
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{model_name}] [{index + 1}/{len(eval_set)}] "
            f"strict={strict_correct} official={repository_correct} "
            f"completed={completed} pred={prediction!r} gold={gold!r}"
        )

    total = len(eval_set)
    elapsed = time.time() - start_time
    summary: dict[str, object] = {
        "ablation": "wo_interactive_coding_paper_aligned",
        "llm": model_name,
        "seed": seed,
        "done": total,
        "total": total,
        "correct": strict_correct_count,
        "accuracy": round(strict_correct_count / total * 100, 2) if total else 0.0,
        "official_correct": official_correct_count,
        "official_accuracy": round(official_correct_count / total * 100, 2)
        if total
        else 0.0,
        "completed": completed_count,
        "completion_rate": round(completed_count / total * 100, 2)
        if total
        else 0.0,
        "elapsed_sec": elapsed,
        "question_ids": [item["id"] for item in eval_set],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished [{model_name}]: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="all")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument(
        "--data_path",
        default="data/ehrsql-ehragent/mimic_iii/valid_preprocessed.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/ablation/wo_interactive_coding_paper_aligned",
    )
    args = parser.parse_args()
    models = (
        ["deepseek-v4-flash-0731", "gpt-oss-120b-openrouter"]
        if args.llm == "all"
        else [args.llm]
    )
    eval_set = select_eval_items(args.data_path, args.num_questions, args.seed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                evaluate_model_wo_interactive_coding,
                model,
                eval_set,
                args.seed,
                args.output_dir,
                args.max_turns,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error executing {model}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
