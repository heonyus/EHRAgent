###
# `prompts_mimic.py` 에서 정의한 프롬프트를 사용하여 지식을 추출
# _MIMIC_IV_SCHEMA을 사용해 LLM에게 현재 DB 구조 설명
###

import time
from src.client import call_llm
from src.log import note
from src.prompts.prompts_mimic3 import RetrKnowledge


def retrieve_knowledge(query, model_name, patience=2, sleep_time=30):
    query_message = RetrKnowledge.format(question=query)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {"role": "user", "content": query_message},
    ]

    while patience > 0:
        patience -= 1
        try:
            response = call_llm(model_name, messages, label="knowledge")
            prediction = response.choices[0].message.content
            if prediction:
                return prediction.strip()
        except Exception as e:
            note(e)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return "Fail to retrieve related knowledge, please try again later."