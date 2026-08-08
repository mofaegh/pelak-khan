from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

APP_NAME = "Pelak-Khan"
APP_VERSION = "0.5.0"
MUTEX_NAME = "Local\\PelakKhanPortable_v1"
DEFAULT_PORT_START = 8765
DEFAULT_PORT_END = 8795


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Read-only application resources bundled by PyInstaller."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parents[2]


def portable_root() -> Path:
    """Writable root of the portable application folder."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return bundle_root()


def configure_environment() -> dict[str, Path]:
    resources = bundle_root()
    root = portable_root()

    # Full portable mode: user content stays beside Pelak-Khan.exe.
    # An advanced user can override the data folder with PELAK_PORTABLE_DATA_DIR.
    data = Path(os.getenv("PELAK_PORTABLE_DATA_DIR", str(root / "data"))).resolve()
    storage = data / "storage"
    backups = data / "backups"
    logs = data / "logs"
    runtime = data / "runtime"

    for directory in (data, storage, backups, logs, runtime):
        directory.mkdir(parents=True, exist_ok=True)

    # Ensure the portable folder is actually writable before starting the API.
    probe = runtime / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "The Pelak-Khan portable folder is not writable. "
            "Extract it to a normal folder such as Desktop, Documents, or another drive."
        ) from exc

    detector = resources / "models" / "runtime" / "detector_v1.pt"
    ocr = resources / "models" / "runtime" / "ocr_v1.pt"
    frontend = resources / "apps" / "frontend"

    os.environ["PELAK_APP_ROOT"] = str(resources)
    os.environ["PELAK_DB_PATH"] = str(data / "pelak_khan.db")
    os.environ["PELAK_STORAGE_ROOT"] = str(storage)
    os.environ["PELAK_BACKUPS_DIR"] = str(backups)
    os.environ["PELAK_DETECTOR_PATH"] = str(detector)
    os.environ["PELAK_OCR_PATH"] = str(ocr)
    os.environ["PELAK_FRONTEND_DIR"] = str(frontend)
    os.environ.setdefault("PELAK_DEVICE", "auto")

    return {
        "resources": resources,
        "root": root,
        "data": data,
        "storage": storage,
        "backups": backups,
        "logs": logs,
        "runtime": runtime,
        "detector": detector,
        "ocr": ocr,
        "frontend": frontend,
    }


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_dir / "pelak-khan.log", encoding="utf-8")
    ]
    if not is_frozen():
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def show_message(title: str, message: str, error: bool = False) -> None:
    if sys.platform == "win32":
        import ctypes

        flags = 0x10 if error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, title, flags)
    else:
        print(f"{title}: {message}", file=sys.stderr if error else sys.stdout)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def choose_port() -> int:
    for port in range(DEFAULT_PORT_START, DEFAULT_PORT_END + 1):
        if _port_is_free(port):
            return port
    raise RuntimeError("No free local port is available for Pelak-Khan.")


def wait_for_health(url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.35)
    raise RuntimeError(f"Pelak-Khan backend did not become ready: {last_error}")


def acquire_single_instance(runtime_file: Path) -> tuple[Any | None, bool]:
    if sys.platform != "win32":
        return None, True

    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    error_already_exists = 183
    if kernel32.GetLastError() == error_already_exists:
        try:
            state = json.loads(runtime_file.read_text(encoding="utf-8"))
            port = int(state["port"])
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            show_message(APP_NAME, "Pelak-Khan is already running. Check the system tray.")
        return handle, False
    return handle, True


def release_mutex(handle: Any | None) -> None:
    if handle and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.ReleaseMutex(handle)
        ctypes.windll.kernel32.CloseHandle(handle)


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


def run_self_test(paths: dict[str, Path]) -> int:
    required = {
        "frontend": paths["frontend"] / "index.html",
        "detector": paths["detector"],
        "ocr": paths["ocr"],
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        show_message(APP_NAME, "Missing application files:\n\n" + "\n".join(missing), error=True)
        return 2

    from apps.backend.main import app

    # Validate the exact Uvicorn configuration used by the portable app.
    # This catches packaging-specific logging/configuration failures before
    # a release is published, without binding a TCP port.
    import uvicorn

    uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=DEFAULT_PORT_START,
        log_level="warning",
        access_log=False,
        log_config=None,
    )

    if app.version != APP_VERSION:
        show_message(
            APP_NAME,
            f"Unexpected backend version: {app.version} (expected {APP_VERSION})",
            error=True,
        )
        return 3
    return 0


def run_desktop() -> int:
    paths = configure_environment()
    setup_logging(paths["logs"])
    logger = logging.getLogger("pelak_khan.desktop")

    runtime_file = paths["runtime"] / "runtime.json"
    mutex_handle, primary = acquire_single_instance(runtime_file)
    if not primary:
        return 0

    server = None
    thread: threading.Thread | None = None
    try:
        code = run_self_test(paths)
        if code:
            return code

        port = choose_port()
        runtime_file.write_text(
            json.dumps({"port": port, "pid": os.getpid(), "version": APP_VERSION}, ensure_ascii=False),
            encoding="utf-8",
        )

        import uvicorn
        from apps.backend.main import app

        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            # Pelak-Khan configures its own file logging above.  Disabling
            # Uvicorn's dictConfig avoids formatter/import issues in the
            # relocatable bundled Python runtime.
            log_config=None,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="pelak-khan-server", daemon=True)
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        wait_for_health(base_url + "/health")
        webbrowser.open(base_url + "/")
        logger.info("Pelak-Khan %s started on %s", APP_VERSION, base_url)

        try:
            import pystray
            from PIL import Image

            tray_icon_path = paths["resources"] / "build" / "windows" / "assets" / "pelak-khan.ico"
            image = Image.open(tray_icon_path)

            def open_app(icon=None, item=None):  # noqa: ANN001
                webbrowser.open(base_url + "/")

            def open_data(icon=None, item=None):  # noqa: ANN001
                open_folder(paths["data"])

            def open_logs(icon=None, item=None):  # noqa: ANN001
                open_folder(paths["logs"])

            def exit_app(icon, item=None):  # noqa: ANN001
                if server is not None:
                    server.should_exit = True
                icon.stop()

            menu = pystray.Menu(
                pystray.MenuItem("Open Pelak-Khan", open_app, default=True),
                pystray.MenuItem("Open portable data", open_data),
                pystray.MenuItem("Open logs", open_logs),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit", exit_app),
            )
            icon = pystray.Icon(APP_NAME, image, f"{APP_NAME} {APP_VERSION}", menu)
            icon.run()
        except Exception as exc:  # noqa: BLE001
            logger.exception("System tray failed: %s", exc)
            show_message(
                APP_NAME,
                "Pelak-Khan is running in your browser. Keep Pelak-Khan.exe open while using the app.",
            )
            while thread.is_alive():
                time.sleep(0.5)

        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pelak-Khan failed to start")
        show_message(APP_NAME, f"Pelak-Khan could not start:\n\n{exc}", error=True)
        return 1
    finally:
        runtime_file.unlink(missing_ok=True)
        release_mutex(mutex_handle)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=not is_frozen())
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--open-data", action="store_true")
    parser.add_argument("--version", action="store_true")
    args, _ = parser.parse_known_args()

    if args.version:
        print(APP_VERSION)
        return 0

    try:
        paths = configure_environment()
        setup_logging(paths["logs"])
        if args.self_test:
            return run_self_test(paths)
        if args.open_data:
            open_folder(paths["data"])
            return 0
        return run_desktop()
    except Exception as exc:  # noqa: BLE001
        show_message(APP_NAME, str(exc), error=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
