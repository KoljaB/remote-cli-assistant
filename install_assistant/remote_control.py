from __future__ import annotations

import json
import shutil
import socket
import socketserver
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .models import CommandResult, CommandTask, ProbeRecord, SystemProfile, WorkerHello
from .session_io import SessionPaths, read_json, write_json
from .system_probe import collect_system_profile
from .worker_launcher import launch_visible_worker


POLL_INTERVAL_SECONDS = 0.05


class SessionCommandBridge:
    def __init__(self, *, session: SessionPaths, worker_process: subprocess.Popen[bytes]) -> None:
        self.session = session
        self.worker_process = worker_process
        self.event_offset = 0

    def execute(
        self,
        task: CommandTask,
        *,
        on_event: callable | None = None,
    ) -> CommandResult:
        write_json(self.session.command_path(task.step_number), task.to_dict())
        result_path = self.session.result_path(task.step_number)
        while True:
            self.event_offset = self._drain_events(self.event_offset, on_event)
            if result_path.exists():
                self.event_offset = self._drain_events(self.event_offset, on_event)
                return CommandResult.from_dict(read_json(result_path))
            if self.worker_process.poll() is not None:
                raise RuntimeError("The controlled CLI window closed unexpectedly.")
            time.sleep(POLL_INTERVAL_SECONDS)

    def close(self, *, state: str, message: str) -> None:
        write_json(self.session.status_path, {"state": state, "message": message})

    def _drain_events(self, offset: int, on_event: callable | None) -> int:
        events_path = self.session.events_path
        if not events_path.exists():
            return offset
        with events_path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if on_event is not None:
                    on_event(record)
            return handle.tell()


class LocalTerminalClient:
    def __init__(self, *, label: str = "AI Controlled CLI") -> None:
        self.session = SessionPaths.create()
        write_json(self.session.status_path, {"state": "running", "message": "CLI assistant session active."})
        self.worker_process = launch_visible_worker(self.session, label=label)
        self.bridge = SessionCommandBridge(session=self.session, worker_process=self.worker_process)
        profile = collect_system_profile()
        self.hello = WorkerHello(
            worker_label=label,
            default_cwd=str(Path.home()),
            system_profile=profile.to_dict(),
        )

    def execute(self, task: CommandTask, *, on_event: callable | None = None) -> CommandResult:
        return self.bridge.execute(task, on_event=on_event)

    def close(self) -> None:
        self.bridge.close(state="stopped", message="CLI assistant session closed.")
        process_exited = False
        try:
            self.worker_process.wait(timeout=2)
            process_exited = True
        except subprocess.TimeoutExpired:
            pass
        if process_exited:
            shutil.rmtree(self.session.root, ignore_errors=True)


class RemoteTerminalClient:
    def __init__(self, *, host: str, port: int, timeout_seconds: int = 15) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    def hello(self) -> WorkerHello:
        record = self._single_request({"action": "hello"})
        if record.get("event") != "hello":
            raise RuntimeError(f"Unexpected hello response: {record}")
        return WorkerHello.from_dict(record["hello"])

    def execute(self, task: CommandTask, *, on_event: callable | None = None) -> CommandResult:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(None)
            sock_file = sock.makefile("rwb")
            _send_record(sock_file, {"action": "execute", "task": task.to_dict()})
            while True:
                record = _read_record(sock_file)
                event_type = record.get("event")
                if event_type == "result":
                    return CommandResult.from_dict(record["result"])
                if event_type == "error":
                    raise RuntimeError(record.get("message", "Remote worker error."))
                if on_event is not None:
                    on_event(record)

    def close(self) -> None:
        return None

    def _single_request(self, payload: dict) -> dict:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(None)
            sock_file = sock.makefile("rwb")
            _send_record(sock_file, payload)
            return _read_record(sock_file)


@dataclass
class _ServerState:
    hello: WorkerHello
    bridge: SessionCommandBridge
    lock: threading.Lock
    shutdown_requested: threading.Event


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RequestHandler(socketserver.StreamRequestHandler):
    server: "_RemoteControlServer"

    def handle(self) -> None:
        try:
            request = _read_record(self.rfile)
            action = request.get("action")
            if action == "hello":
                _send_record(self.wfile, {"event": "hello", "hello": self.server.state.hello.to_dict()})
                return
            if action == "execute":
                task = CommandTask.from_dict(request["task"])
                with self.server.state.lock:
                    result = self.server.state.bridge.execute(task, on_event=lambda record: _send_record(self.wfile, record))
                _send_record(self.wfile, {"event": "result", "result": result.to_dict()})
                return
            if action == "shutdown":
                self.server.state.shutdown_requested.set()
                self.server.state.bridge.close(state="stopped", message="Remote worker server stopped.")
                _send_record(self.wfile, {"event": "shutdown", "ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            _send_record(self.wfile, {"event": "error", "message": f"Unknown action: {action}"})
        except Exception as exc:
            _send_record(self.wfile, {"event": "error", "message": str(exc)})


class _RemoteControlServer(_ThreadingTCPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[_RequestHandler], state: _ServerState) -> None:
        super().__init__(server_address, handler_class)
        self.state = state


def run_remote_worker_server(*, listen_host: str, port: int) -> int:
    session = SessionPaths.create()
    write_json(session.status_path, {"state": "running", "message": "Remote CLI worker is active."})
    worker_process = launch_visible_worker(session, label="AI Controlled Remote CLI")
    profile = collect_system_profile()
    hello = WorkerHello(
        worker_label=f"{socket.gethostname()}:{port}",
        default_cwd=str(Path.home()),
        system_profile=profile.to_dict(),
    )
    bridge = SessionCommandBridge(session=session, worker_process=worker_process)
    state = _ServerState(
        hello=hello,
        bridge=bridge,
        lock=threading.Lock(),
        shutdown_requested=threading.Event(),
    )
    server = _RemoteControlServer((listen_host, port), _RequestHandler, state)

    print(f"Remote CLI worker listening on {listen_host}:{port}", flush=True)
    print("A visible terminal window should now be open on this machine.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close(state="stopped", message="Remote worker server stopped.")
        server.server_close()
        process_exited = False
        try:
            worker_process.wait(timeout=2)
            process_exited = True
        except subprocess.TimeoutExpired:
            pass
        if process_exited:
            shutil.rmtree(session.root, ignore_errors=True)
    return 0


def system_profile_from_hello(hello: WorkerHello) -> SystemProfile:
    payload = dict(hello.system_profile)
    payload["records"] = [ProbeRecord(**record) for record in payload.get("records", [])]
    return SystemProfile(**payload)


def parse_connect_target(value: str) -> tuple[str, int]:
    host, _, raw_port = value.partition(":")
    if not host or not raw_port:
        raise ValueError("Expected HOST:PORT for --connect.")
    return host, int(raw_port)


def _send_record(handle, payload: dict) -> None:
    handle.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
    handle.flush()


def _read_record(handle) -> dict:
    line = handle.readline()
    if not line:
        raise RuntimeError("Connection closed unexpectedly.")
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return json.loads(line)
