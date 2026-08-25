import json
import re

from src.agent_tools import get_tools
from src.client import call_llm
from src.log import block, note
from src.memory.debugger import error_debugger
from src.memory.few_shot import retrieve_examples, seed_memory
from src.memory.knowledge import retrieve_knowledge
from src.prompts.prompts_mimic3 import (
    EHRAgent_4Shots_Knowledge,
    EHRAgent_Message_Prompt,
)
from src.tools.python_excute import run_code


def extract_tool_code(message):
    """tool call 응답에서 코드를 꺼낸다. (JSON이 깨져 있으면 정규식으로 복구)"""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return "", None
    tc = tool_calls[0]
    raw = tc.function.arguments or "{}"
    try:
        cell = json.loads(raw).get("cell", "")
    except json.JSONDecodeError:
        match = re.search(r'"cell"\s*:\s*"(.*)"', raw, re.DOTALL)
        cell = match.group(1) if match else ""
    return cell, tc.id


def append_feedback(messages, message, call_id, feedback):
    """다음 턴을 위해 '이번 시도 + 실행 결과'를 대화에 기록한다."""
    messages.append({"role": "assistant", "content": message.content or ""})
    if call_id:
        messages.append({"role": "tool", "tool_call_id": call_id, "content": feedback})
    else:
        messages.append({"role": "user", "content": feedback})


def run_question(question, model_name, memory=None, max_turns=10):
    block("question", "{}\nmodel: {}".format(question, model_name))

    knowledge = retrieve_knowledge(question, model_name)
    block("knowledge", knowledge)

    if memory is None:
        memory = seed_memory(EHRAgent_4Shots_Knowledge)

    examples = retrieve_examples(question, memory)
    note("few-shot  {} examples".format(examples.count("Question:")))
    prompt = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=question)
    messages = [
        {"role": "system", "content": "For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code."},
        {"role": "user", "content": prompt},
    ]

    last_result = None
    last_code = None
    for turn in range(max_turns):

        message = call_llm(
            model_name, messages, tools=get_tools(),
            label="turn {}".format(turn),
        ).choices[0].message

        content = message.content or ""
        cell, call_id = extract_tool_code(message)

        if cell:
            block("turn {}  code".format(turn), cell.strip(), kind="code")
            last_code = cell
            result = run_code(cell)
            last_result = result
            if "error" in result or "Error" in result:
                block("turn {}  error".format(turn), result, kind="err")
                reason = error_debugger(question, cell, result, model_name)
                result = result + "\nPotential Reasons: " + str(reason)
                block("turn {}  debugger".format(turn), reason)
            append_feedback(messages, message, call_id, result)
        else:
            messages.append({"role": "assistant", "content": content})

        if content.rstrip().endswith("TERMINATE"): # (TERMINATE로 끝날 때)
            block("answer", last_result, kind="ok")
            return {"knowledge": knowledge, "code": last_code, "answer": last_result, "terminated": True}


    block("answer", last_result, kind="err")
    return {"knowledge": knowledge, "code": last_code, "answer": last_result, "terminated": False}
