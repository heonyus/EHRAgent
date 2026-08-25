import time

from src.client import call_llm
from src.log import note
from src.prompts.prompts_mimic3 import CodeDebugger


def error_debugger(question, code, error_info, model_name, patience=2, sleep_time=30):
    query_message = CodeDebugger.format(
        question=question,
        code=code,
        error_info=error_info,
    )
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors.",
        },
        {"role": "user", "content": query_message},
    ]
    while patience > 0:
        patience -= 1
        try:
            response = call_llm(model_name, messages, label="debugger")
            prediction = response.choices[0].message.content
            if prediction:
                return prediction.strip()
        except Exception as e:
            note(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return "Fail to diagnose the reasons to the errors."