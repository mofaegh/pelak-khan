from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "Pelak-Khan"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def show_error(message: str) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
    else:
        print(f"{APP_NAME}: {message}", file=sys.stderr)


def main() -> int:
    root = app_root()
    runtime = root / "runtime" / "python"
    python_console = runtime / "python.exe"
    python_windowed = runtime / "pythonw.exe"

    if not python_console.exists() or not python_windowed.exists():
        show_error(
            "Portable Python runtime is missing.\n\n"
            "Please extract the complete Pelak-Khan ZIP before running the application."
        )
        return 2

    env = os.environ.copy()
    env["PYTHONHOME"] = str(runtime)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    env["PYTHONNOUSERSITE"] = "1"
    env["PELAK_PORTABLE_ROOT"] = str(root)

    passthrough = sys.argv[1:]
    synchronous = any(arg in {"--self-test", "--version", "--open-data"} for arg in passthrough)
    child_python = python_console if synchronous else python_windowed
    command = [str(child_python), "-m", "apps.desktop.launcher", *passthrough]

    try:
        if synchronous:
            return subprocess.call(command, cwd=str(root), env=env)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            creationflags=creationflags,
            close_fds=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        show_error(f"Pelak-Khan could not start.\n\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
