import argparse
import atexit
import json
import os
import socket
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="HomeEase local care system")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stop", action="store_true", help="Gracefully stop the local HomeEase instance")
    args = parser.parse_args()
    data_directory = Path(os.environ.get("HOMEEASE_SYSTEM_DATA", str(Path(__file__).resolve().parent / "runtime")))
    stop_file = data_directory / "stop.request"
    if args.stop:
        data_directory.mkdir(parents=True, exist_ok=True)
        stop_file.write_text("stop", encoding="ascii")
        print("Stop requested; models, streams and database will be released.")
        return
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("HOMEEASE_PASSWORD"):
        parser.error("LAN access requires HOMEEASE_PASSWORD; username is caregiver. Use HTTPS for browser devices.")
    data_directory.mkdir(parents=True, exist_ok=True)
    instance_lock = (data_directory / "server.lock").open("a+b")
    if instance_lock.tell() == 0:
        instance_lock.write(b"0")
        instance_lock.flush()
    instance_lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(instance_lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        instance_lock.close()
        address_file = data_directory / "server-address.json"
        try:
            address = json.loads(address_file.read_text(encoding="utf-8"))["url"]
        except (OSError, ValueError, KeyError):
            address = "initializing"
        print(f"HomeEase is already running: {address}")
        return
    atexit.register(instance_lock.close)
    port = args.port
    for candidate in range(port, port + 50):
        try:
            with socket.socket() as probe:
                probe.bind((args.host, candidate))
            port = candidate
            break
        except OSError:
            continue
    else:
        parser.error("No free port in the selected range")
    print(f"HomeEase: http://{args.host}:{port}", flush=True)
    (data_directory / "server-address.json").write_text(json.dumps({"url": f"http://{args.host}:{port}", "pid": os.getpid()}), encoding="utf-8")
    import uvicorn
    stop_file.unlink(missing_ok=True)
    server = uvicorn.Server(uvicorn.Config("care.main:app", host=args.host, port=port, workers=1, log_level="info"))
    finished = threading.Event()

    def watch_stop():
        while not finished.wait(.5):
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
                server.should_exit = True
                return

    threading.Thread(target=watch_stop, name="homeease-stop", daemon=True).start()
    try:
        server.run()
    finally:
        finished.set()


if __name__ == "__main__":
    main()
