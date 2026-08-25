import io
from contextlib import redirect_stdout

from src.tools.tabtools_mimic3 import (
    db_loader,
    data_filter,
    get_value,
    sql_interpreter,
    date_calculator,
)

def format_error(e, cell):
    # 원본 github/EhrAgent/ehragent/toolset_high.py 와 같은 에러 문장.
    code = cell
    if "SyntaxError" in str(repr(e)):
        error_line = str(repr(e))
        error_type = error_line.split("(")[0]
        error_message = error_line.split(",")[0].split("(")[1]
        error_line = error_line.split('"')[1]
    elif "KeyError" in str(repr(e)):
        code = code.split("\n")
        key = str(repr(e)).split("'")[1]
        error_type = str(repr(e)).split("(")[0]
        error_line = ""
        for line in code:
            if key in line:
                error_line = line
        error_message = str(repr(e))
    elif "TypeError" in str(repr(e)):
        error_type = str(repr(e)).split("(")[0]
        error_message = str(e)
        names = {
            "get_value": "GetValue",
            "data_filter": "FilterDB",
            "db_loader": "LoadDB",
            "sql_interpreter": "SQLInterpreter",
            "date_calculator": "Calendar",
        }
        error_key = ""
        for key, alias in names.items():
            if key in error_message:
                error_message = error_message.replace(key, alias)
                error_key = alias
        error_line = ""
        for line in code.split("\n"):
            if error_key and error_key in line:
                error_line = line
    else:
        error_type = ""
        error_message = str(repr(e)).split("('")[-1].split("')")[0]
        error_line = ""
    if error_type != "" and error_line != "":
        error_info = '{}: {}. The error messages occur in the code line "{}".'.format(
            error_type, error_message, error_line
        )
    else:
        error_info = "Error: {}.".format(error_message)
    return error_info + "\nPlease make modifications accordingly and make sure the rest code works well with the modification."

def run_code(cell):
    ns = {
        "LoadDB": db_loader,
        "FilterDB": data_filter,
        "GetValue": get_value,
        "SQLInterpreter": sql_interpreter,
        "Calendar": date_calculator,
        "answer": 0,
    }
    try:
        with redirect_stdout(io.StringIO()):
            exec(cell, ns)
    except Exception as e:
        return format_error(e, cell)

    code = "\n".join(
        line for line in cell.split("\n")
        if line.strip() and not line.strip().startswith("#")
    )
    if "answer" not in code.split("\n")[-1]:
        return "Please save the answer to the question in the variable 'answer'."
    return str(ns["answer"])