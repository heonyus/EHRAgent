"""
EHRAgent 4-Component Ablation Evaluation Pipeline
=================================================
이 스크립트는 EHRAgent의 4대 핵심 컴포넌트 어블레이션을 병렬로 실행하고 평가합니다:
  1. no_knowledge : Knowledge Retrieval 제외 (knowledge="")
  2. no_memory    : Few-shot Memory 제외 (examples="", 0-shot)
  3. no_debugger  : Error Debugger 제외 (에러 발생 시 디버거 분석 없이 raw error만 피드백)
  4. no_feedback  : Multi-turn Feedback 제외 (단일 턴 max_turns=1 코드 실행)

※ 참고 (w/o Interactive Coding):
  - 자연어 CoT 계획 + 단계별 Tool 호출 방식의 진짜 `w/o interactive coding` 실험은
    완전히 독립된 전용 스크립트인 `ablation_wo_interactive_coding.py`에서 실행됩니다.
"""

import argparse
import json
import os
import random
import time
from typing import Optional

from src.agent import append_feedback, extract_tool_code
from src.agent_tools import get_tools
from src.client import call_llm
from src.log import block, note
from src.memory.debugger import error_debugger
from src.memory.few_shot import parse_shots, retrieve_examples, seed_memory
from src.memory.knowledge import retrieve_knowledge
from src.prompts.prompts_mimic3 import (
    EHRAgent_4Shots_Knowledge,
    EHRAgent_Message_Prompt,
)
from src.tools.python_excute import run_code


def judge(pred: str, ans: str) -> bool:
    old_flag = True
    if ans not in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans in ["False", "false", "No", "no", "None", "none"]:
        ans = "0"
    elif ans in ["True", "true", "Yes", "yes"]:
        ans = "1"
    if ", " in ans:
        ans_list = ans.split(", ")
    elif ans.endswith(".0"):
        ans_list = [ans[:-2]]
    elif isinstance(ans, list):
        ans_list = ans
    else:
        ans_list = [ans]

    new_flag = True
    for a in ans_list:
        if a not in pred:
            new_flag = False
            break
    return old_flag or new_flag


def load_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def run_question_ablation(
    question: str,
    model_name: str,
    ablation_type: str,
    memory=None,
    max_turns: int = 10,
):
    """
    4대 Ablation 실험별 실행 로직:
    1. no_knowledge : Knowledge Retrieval 제외 (knowledge="")
    2. no_memory    : Few-shot Memory 제외 (examples="", memory 미사용)
    3. no_debugger  : Error Debugger 제외 (에러 발생 시 디버거 분석 없이 raw error만 피드백)
    4. no_feedback  : Execution Feedback 제외 (단일 턴 max_turns=1 실행)
    """
    block("question", f"[{ablation_type}] {question}\nmodel: {model_name}")

    # 1. Knowledge Retrieval 설정
    if ablation_type == "no_knowledge":
        knowledge = ""
        note("Ablation: Knowledge Retrieval disabled")
    else:
        knowledge = retrieve_knowledge(question, model_name)
        block("knowledge", knowledge)

    # 2. Example Memory 설정
    if ablation_type == "no_memory":
        examples = ""
        note("Ablation: Few-shot Memory disabled (0-shot)")
    else:
        if memory is None:
            memory = seed_memory(EHRAgent_4Shots_Knowledge)
        examples = retrieve_examples(question, memory)
        note(f"few-shot: {examples.count('Question:')} examples")

    prompt = EHRAgent_Message_Prompt.format(
        examples=examples, knowledge=knowledge, question=question
    )
    messages = [
        {
            "role": "system",
            "content": (
                "For coding tasks, only use the functions you have been provided with. "
                "Reply TERMINATE when the task is done. Save the answers to the questions "
                "in the variable 'answer'. Please only generate the code."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # 4. Feedback / 턴 수 제어
    effective_turns = max_turns
    # effective_turns = 1 if ablation_type == "no_feedback" else max_turns  # (w/o Interactive Coding은 ablation_wo_interactive_coding.py로 완전 분리됨)

    last_result = None
    last_code = None

    for turn in range(effective_turns):
        message = call_llm(
            model_name,
            messages,
            tools=get_tools(),
            label=f"turn {turn} [{ablation_type}]",
        ).choices[0].message

        content = message.content or ""
        cell, call_id = extract_tool_code(message)

        if cell:
            block(f"turn {turn} code", cell.strip(), kind="code")
            last_code = cell
            result = run_code(cell)
            last_result = result

            if "error" in result or "Error" in result:
                block(f"turn {turn} error", result, kind="err")
                # 3. Error Debugger 제어
                if ablation_type != "no_debugger":
                    reason = error_debugger(question, cell, result, model_name)
                    result = result + "\nPotential Reasons: " + str(reason)
                    block(f"turn {turn} debugger", reason)
                else:
                    note("Ablation: Error Debugger disabled (raw error only)")

            append_feedback(messages, message, call_id, result)
        else:
            messages.append({"role": "assistant", "content": content})

        if content.rstrip().endswith("TERMINATE"):
            block("answer", last_result, kind="ok")
            return {
                "knowledge": knowledge,
                "code": last_code,
                "answer": last_result,
                "terminated": True,
            }

    # max_turns 초과 또는 no_feedback 완료 시
    block("answer", last_result, kind="err" if last_result is None else "info")
    return {
        "knowledge": knowledge,
        "code": last_code,
        "answer": last_result,
        "terminated": False,
    }


def run_ablation_experiment(
    ablation_type: str,
    llm: str,
    num_questions: int,
    start_id: int,
    seed: int,
    data_path: str,
    logs_base_path: str = "logs/ablation",
):
    print(f"\n=======================================================")
    print(f"🚀 Starting Ablation Experiment: [{ablation_type}]")
    print(f"Model: {llm} | Questions: {start_id} ~ {num_questions} | Seed: {seed}")
    print(f"=======================================================\n")

    random.seed(seed)
    contents = load_items(data_path)
    random.shuffle(contents)

    if num_questions == -1 or num_questions > len(contents):
        num_questions = len(contents)

    log_dir = os.path.join(logs_base_path, ablation_type)
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "{id}.txt")

    memory = parse_shots(EHRAgent_4Shots_Knowledge)
    n_correct = 0
    start_time = time.time()
    results_path = os.path.join(log_dir, f"results_{llm}_{seed}.jsonl")
    summary_path = os.path.join(log_dir, f"summary_{llm}_{seed}.json")

    # 결과 파일 초기화 (새로 시작하는 경우)
    if start_id == 0:
        open(results_path, "w", encoding="utf-8").close()

    for i in range(start_id, num_questions):
        item = contents[i]
        question = item["template"]
        answer = item["answer"]
        gold_raw = answer

        out = run_question_ablation(
            question,
            llm,
            ablation_type=ablation_type,
            memory=memory if ablation_type != "no_memory" else None,
        )

        if isinstance(answer, list):
            answer_str = ", ".join(str(a) for a in answer)
        else:
            answer_str = str(answer)

        logs_string = [str(question), str(gold_raw)]
        if out.get("code"):
            cell = str(out["code"]).replace("\\", "\\\\").replace('"', '\\"')
            logs_string.append('"cell": "' + cell + '"\n}')
        if out.get("answer") is not None:
            logs_string.append(str(out["answer"]))
        if out.get("terminated"):
            logs_string.append("TERMINATE")
        logs_string.append("Ground-Truth Answer ---> " + answer_str)

        joined = "\n----------------------------------------------------------\n".join(
            logs_string
        )
        with open(file_path.format(id=item["id"]), "w", encoding="utf-8") as f:
            f.write(joined)

        # 판정 로직
        # if ablation_type == "no_feedback":
        #     pred_text = str(out.get("answer") or "")
        #     result = judge(pred_text, answer_str)
        # else:
        if not out.get("terminated"):
            result = False
        elif '"cell": "' in joined:
            last_code_end = joined.rfind('"\n}')
            prediction = joined[last_code_end : joined.rfind("TERMINATE")]
            result = judge(prediction, answer_str)
        else:
            last_code_end = joined.rfind("Solution:")
            prediction = joined[last_code_end : joined.rfind("TERMINATE")]
            result = judge(prediction, answer_str)

        n_correct += int(bool(result))
        print(
            f"[{ablation_type}] [{i + 1}/{num_questions}] correct: {result} | pred: {out.get('answer')} | gold: {answer_str}"
        )

        row = {
            "i": i,
            "id": item["id"],
            "ablation": ablation_type,
            "question": question,
            "gold": answer_str,
            "pred": out.get("answer"),
            "correct": bool(result),
            "terminated": bool(out.get("terminated")),
            "code": out.get("code"),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if result and ablation_type != "no_memory":
            memory.append(
                {
                    "question": question,
                    "knowledge": out.get("knowledge", ""),
                    "code": out.get("code", ""),
                }
            )

        elapsed = time.time() - start_time
        n_done = i - start_id + 1
        summary = {
            "ablation": ablation_type,
            "llm": llm,
            "seed": seed,
            "done": n_done,
            "total": num_questions - start_id,
            "correct": n_correct,
            "score": f"{n_correct}/{n_done}",
            "accuracy": round(n_correct / n_done * 100, 2) if n_done > 0 else 0,
            "elapsed_sec": elapsed,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{ablation_type}] Final Score: {n_correct}/{num_questions - start_id}")
    print(f"Summary saved to: {summary_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="EHRAgent Ablation Experiments")
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["no_knowledge", "no_memory", "no_debugger", "all"],
        # choices=["no_knowledge", "no_memory", "no_debugger", "no_feedback", "all"],  # no_feedback (1턴 실행) 비활성화
        help="Ablation type to run",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-oss-120b-openrouter",
        help="LLM model name from config/models.yaml",
    )
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ehrsql-ehragent/mimic_iii/valid_preprocessed.jsonl",
    )
    parser.add_argument("--logs_path", type=str, default="logs/ablation")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run ablation experiments in parallel processes (default: True for all)",
    )
    args = parser.parse_args()

    # 3대 핵심 컴포넌트 어블레이션 (w/o Interactive Coding 관련 no_feedback은 주석 처리)
    experiments = (
        ["no_knowledge", "no_memory", "no_debugger"]
        # ["no_knowledge", "no_memory", "no_debugger", "no_feedback"]
        if args.ablation == "all"
        else [args.ablation]
    )

    all_summaries = {}

    if len(experiments) > 1 and args.parallel:
        import concurrent.futures

        print(f"🚀 Running {len(experiments)} ablation experiments in PARALLEL...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(experiments)) as executor:
            futures = {
                executor.submit(
                    run_ablation_experiment,
                    exp,
                    args.llm,
                    args.num_questions,
                    args.start_id,
                    args.seed,
                    args.data_path,
                    args.logs_path,
                ): exp
                for exp in experiments
            }
            for future in concurrent.futures.as_completed(futures):
                exp = futures[future]
                try:
                    all_summaries[exp] = future.result()
                except Exception as e:
                    print(f"❌ Error in experiment [{exp}]: {e}")
    else:
        for exp in experiments:
            summary = run_ablation_experiment(
                ablation_type=exp,
                llm=args.llm,
                num_questions=args.num_questions,
                start_id=args.start_id,
                seed=args.seed,
                data_path=args.data_path,
                logs_base_path=args.logs_path,
            )
            all_summaries[exp] = summary

    # 종합 리포트 저장
    overall_path = os.path.join(args.logs_path, f"overall_summary_{args.llm}_{args.seed}.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print("All Ablation Experiments Completed!")
    print(f"Overall summary saved to: {overall_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
