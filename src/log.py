"""Stdout-only run log. One block per step, no duplicate dumps."""

import sys

_TTY = sys.stdout.isatty()
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"

_KIND = {
    "info": _CYAN,
    "ok": _GREEN,
    "err": _RED,
    "code": _YELLOW,
    "llm": _MAGENTA,
}


def _paint(color, text):
    if not _TTY:
        return text
    return f"{color}{text}{_RESET}"


def note(text):
    print(_paint(_DIM, str(text)), flush=True)


def block(title, body="", kind="info"):
    color = _KIND.get(kind, _CYAN)
    bar = "─" * 64
    print(flush=True)
    print(_paint(color, bar), flush=True)
    print(_paint(_BOLD + color, " " + title), flush=True)
    print(_paint(color, bar), flush=True)
    if body is None:
        return
    text = str(body).rstrip()
    if text:
        print(text, flush=True)
