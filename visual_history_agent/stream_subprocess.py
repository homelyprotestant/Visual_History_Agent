"""Run subprocesses with live stdout for notebooks / Colab."""

from __future__ import annotations

import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    keep_lines: int = 80,
) -> None:
    """Run ``cmd``, printing each output line immediately (unbuffered child).

    Jupyter/Colab often hide child stdout until exit when the child is fully
    buffered (non-TTY). Force ``python -u`` + ``PYTHONUNBUFFERED`` and tee
    line-by-line into the parent with ``flush=True``.

    On failure, the exception includes the last ``keep_lines`` of output so the
    real agent error is visible even when the notebook scrollback is noisy.
    """
    command = [str(part) for part in cmd]
    exe = Path(command[0]).name.lower() if command else ""
    if exe.startswith("python") and "-u" not in command[1:3]:
        command.insert(1, "-u")

    merged = os.environ.copy()
    if env:
        merged.update(dict(env))
    merged["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=max(1, int(keep_lines)))
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    code = proc.wait()
    if code != 0:
        detail = "".join(tail).rstrip()
        message = f"command failed (exit {code}): {' '.join(command)}"
        if detail:
            message = f"{message}\n--- last output ---\n{detail}"
        raise RuntimeError(message)
