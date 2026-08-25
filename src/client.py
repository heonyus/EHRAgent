from dotenv import load_dotenv
from litellm import completion
import litellm
litellm.suppress_debug_info = True

from .log import block, note
from .model_config import get_model_config

load_dotenv()


def call_llm(model_name, messages, tools=None, tool_choice=None, label=""):
    tag = "llm  {}".format(model_name)
    if label:
        tag += "  {}".format(label)
    tag += "  messages={}".format(len(messages))
    note(tag)

    model_config = get_model_config(model_name)
    response = completion(
        model=model_config["model"],
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=model_config.get("temperature", 0),
        extra_body=model_config["extra_body"],
        timeout=180,
    )
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        block("reasoning", reasoning.strip(), kind="llm")

    return response