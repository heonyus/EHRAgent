PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "run the entire code and return the execution result. Only generate the code.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell": {
                    "type": "string",
                    "description": "Valid Python code to execute.",
                },
            },
            "required": ["cell"],
        },
    },
}


def get_tools():
    return [PYTHON_TOOL]