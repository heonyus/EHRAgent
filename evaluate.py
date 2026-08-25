import argparse
import json
import os
import random
import time

from src.agent import run_question
from src.memory.few_shot import parse_shots
from src.prompts.prompts_mimic3 import EHRAgent_4Shots_Knowledge


def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(", ")
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return old_flag or new_flag


def load_items(path):
    with open(path) as f:
        text = f.read().strip()
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def set_seed(seed):
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama-3.3-70b")
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ehrsql-ehragent/mimic_iii/valid_preprocessed.jsonl",
    )
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)

    contents = load_items(args.data_path)
    random.shuffle(contents)
    if args.num_questions == -1:
        args.num_questions = len(contents)

    log_dir = os.path.join(args.logs_path, str(args.num_shots))
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "{id}.txt")

    # 원본: 프로세스 RAM 리스트. sqlite seed_memory 쓰지 않음.
    memory = parse_shots(EHRAgent_4Shots_Knowledge)

    n_correct = 0
    start_time = time.time()
    results_path = os.path.join(log_dir, "results_{}_{}.jsonl".format(args.llm, args.seed))
    summary_path = os.path.join(log_dir, "summary_{}_{}.json".format(args.llm, args.seed))
    open(results_path, "w").close()

    for i in range(args.start_id, args.num_questions):
        item = contents[i]
        question = item["template"]
        answer = item["answer"]
        gold_raw = answer

        out = run_question(question, args.llm, memory=memory)
        if type(answer) == list:
            answer = ", ".join(answer)

        logs_string = [str(question), str(gold_raw)]
        if out.get("code"):
            cell = str(out["code"]).replace("\\", "\\\\").replace('"', '\\"')
            logs_string.append('"cell": "' + cell + '"\n}')
        if out.get("answer") is not None:
            logs_string.append(str(out["answer"]))
        if out.get("terminated"):
            logs_string.append("TERMINATE")
        logs_string.append("Ground-Truth Answer ---> " + answer)

        joined = "\n----------------------------------------------------------\n".join(
            logs_string
        )
        with open(file_path.format(id=item["id"]), "w") as f:
            f.write(joined)
        if not out.get("terminated"):
            result = False
        elif '"cell": "' in joined:
            last_code_end = joined.rfind('"\n}')
            prediction = joined[last_code_end:joined.rfind("TERMINATE")]
            result = judge(prediction, answer)
        else:
            last_code_end = joined.rfind("Solution:")
            prediction = joined[last_code_end:joined.rfind("TERMINATE")]
            result = judge(prediction, answer)
        n_correct += int(bool(result))
        print("[{}/{}] correct: {}  pred: {}  gold: {}".format(
            i + 1, args.num_questions, result, out.get("answer"), answer
        ))

        row = {
            "i": i,
            "id": item["id"],
            "question": question,
            "gold": answer,
            "pred": out.get("answer"),
            "correct": bool(result),
            "terminated": bool(out.get("terminated")),
            "code": out.get("code"),
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if result:
            memory.append({
                "question": question,
                "knowledge": out["knowledge"],
                "code": out["code"],
            })

        elapsed = time.time() - start_time
        n_done = i - args.start_id + 1
        summary = {
            "llm": args.llm,
            "seed": args.seed,
            "num_shots": args.num_shots,
            "done": n_done,
            "total": args.num_questions - args.start_id,
            "correct": n_correct,
            "score": "{}/{}".format(n_correct, n_done),
            "elapsed_sec": elapsed,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Time elapsed: ", time.time() - start_time)
    print("score: {}/{}".format(n_correct, args.num_questions - args.start_id))
    print("results:", results_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
