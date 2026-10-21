"""Persistent Python client for the Pi agent JSONL RPC protocol.

No Ollama or physical-tool client lives here: Pi owns model calls and tools.
RPC stdout contains protocol records; never forward raw stdout to speech.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Sequence
from uuid import uuid4

HERE = Path(__file__).resolve().parent
TOOLS = "detect_motion,observe_scene,check_presence,query_memory"
_EOF = object()


class PiRPCError(RuntimeError):
    """Pi could not return a complete, successful agent response."""


@dataclass(frozen=True)
class PiTurn:
    text: str
    elapsed_s: float
    tools: list[str] = field(default_factory=list)
    tool_seconds: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


class PiRPCClient:
    """One live Pi process and one sequential prompt at a time.

    A dedicated reader thread continuously drains stdout. This avoids deadlock
    on large tool events and makes startup/turn deadlines enforceable. Any
    failed/aborted/truncated turn raises instead of being sent to Piper.
    """

    def __init__(
        self,
        *,
        model: str = "ollama/gemma4:e2b-8k",
        cwd: Path = HERE,
        continue_session: bool = False,
        no_session: bool = False,
        timeout_s: float = 180.0,
        startup_timeout_s: float = 45.0,
        command: Sequence[str] | None = None,
    ) -> None:
        if no_session and continue_session:
            raise ValueError("Cannot continue an in-memory session")
        self.cwd = Path(cwd)
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.argv = list(command) if command is not None else [
            "pi", "--mode", "rpc", "--model", model,
            "--tools", TOOLS, "--no-mcp",
            *(["--continue"] if continue_session else []),
            *(["--no-session"] if no_session else []),
        ]
        self._proc: subprocess.Popen[bytes] | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._failed = False

    def __enter__(self) -> "PiRPCClient":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._failed:
            raise PiRPCError("Pi RPC failed during a previous turn; restart the client explicitly")
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self.argv, cwd=str(self.cwd), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=None, bufsize=0,
            )
        except OSError as exc:
            raise PiRPCError(f"Cannot start Pi RPC: {exc}") from exc
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()
        try:
            # Handshake verifies that the RPC process has initialized.
            command_id = self._send("get_state")
            response = self._await_response(command_id, time.monotonic() + self.startup_timeout_s)
            if not response.get("success"):
                raise PiRPCError(f"Pi startup failed: {response.get('error')}")
        except Exception:
            self.close()
            raise

    def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:  # binary LF framing, not Unicode splitlines()
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    self._queue.put(PiRPCError(f"Non-JSON RPC stdout: {exc}"))
                    continue
                self._queue.put(record)
        finally:
            self._queue.put(_EOF)

    def _send(self, kind: str, **fields: Any) -> str:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise PiRPCError("Pi RPC process is no longer running")
        request_id = uuid4().hex
        command = {"id": request_id, "type": kind, **fields}
        try:
            proc.stdin.write(json.dumps(command, ensure_ascii=False).encode("utf-8") + b"\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise PiRPCError("Cannot write to Pi RPC") from exc
        return request_id

    def _next(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PiRPCError("Pi RPC timed out (process will be stopped)")
        try:
            record = self._queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise PiRPCError("Pi RPC timed out (process will be stopped)") from exc
        if record is _EOF:
            code = self._proc.poll() if self._proc else None
            raise PiRPCError(f"Pi RPC closed stdout unexpectedly (exit={code})")
        if isinstance(record, Exception):
            raise record
        if not isinstance(record, dict):
            raise PiRPCError("Unexpected non-object RPC record")
        # Dialog requests require a separate UI implementation. Fail rather than hang.
        if record.get("type") == "extension_ui_request" and record.get("method") in {
            "select", "confirm", "input", "editor"
        }:
            raise PiRPCError(f"Pi extension requires unsupported RPC dialog: {record.get('method')}")
        return record

    def _await_response(self, request_id: str, deadline: float) -> dict[str, Any]:
        while True:
            record = self._next(deadline)
            if record.get("type") == "response" and record.get("id") == request_id:
                return record

    def ask(self, question: str) -> PiTurn:
        if not question.strip():
            raise ValueError("Cannot send an empty question")
        if self._proc is None:
            self.start()
        deadline = time.monotonic() + self.timeout_s
        start = time.perf_counter()
        request_id = self._send("prompt", message=question)
        accepted = False
        settled = False
        assistant_message: dict[str, Any] | None = None
        tools: list[str] = []
        tool_seconds: dict[str, float] = {}
        input_tokens = output_tokens = 0
        try:
            while not (accepted and settled):
                record = self._next(deadline)
                kind = record.get("type")
                if kind == "response" and record.get("id") == request_id:
                    if not record.get("success"):
                        raise PiRPCError(f"Pi rejected prompt: {record.get('error')}")
                    disposition = record.get("data", {}).get("disposition")
                    if disposition != "started":
                        raise PiRPCError(f"Expected a started agent run, got: {disposition}")
                    accepted = True
                elif kind == "message_end":
                    msg = record.get("message") or {}
                    if msg.get("role") == "assistant":
                        usage = msg.get("usage") or {}
                        input_tokens += usage.get("input", 0) or 0
                        output_tokens += usage.get("output", 0) or 0
                        assistant_message = msg
                elif kind == "tool_execution_start":
                    tools.append(str(record.get("toolName", "unknown")))
                elif kind == "tool_execution_end":
                    if record.get("isError"):
                        # A tool error may be handled by Pi; preserve a trace of it.
                        print(f"[pi_rpc] tool error: {record.get('toolName')}", file=sys.stderr)
                    name = str(record.get("toolName", "unknown"))
                    if isinstance(record.get("durationMs"), (int, float)):
                        tool_seconds[name] = tool_seconds.get(name, 0) + record["durationMs"] / 1000
                elif kind == "agent_settled":
                    settled = True
                    if record.get("aborted"):
                        raise PiRPCError("Pi agent run was aborted")
                elif kind == "extension_error":
                    print(f"[pi_rpc] extension error: {record.get('error')}", file=sys.stderr)
            if assistant_message is None:
                raise PiRPCError("Pi did not return a final assistant message")
            reason = assistant_message.get("stopReason")
            if reason != "stop":
                raise PiRPCError(f"Pi did not finish normally (stopReason={reason})")
            content = assistant_message.get("content") or []
            answer = "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            if not answer:
                raise PiRPCError("Pi returned no final answer text")
            return PiTurn(answer, round(time.perf_counter() - start, 3), tools,
                          tool_seconds, input_tokens, output_tokens)
        except Exception:
            # Never silently start a fresh session after a failed run; the new
            # process could lose context while sounding conversationally fluent.
            self._failed = True
            self.close()
            raise

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        finally:
            if proc.stdout:
                proc.stdout.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11 persistent Pi RPC text smoke test")
    parser.add_argument("--model", default="ollama/gemma4:e2b-8k")
    parser.add_argument("--continue-session", action="store_true")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--prompt", help="Run one question and exit; omit for repeated text chat")
    args = parser.parse_args()
    with PiRPCClient(model=args.model, continue_session=args.continue_session,
                     no_session=args.no_session, timeout_s=args.timeout) as client:
        print("[pi_rpc] ready", flush=True)
        while True:
            if args.prompt is not None:
                question = args.prompt
            else:
                try:
                    question = input("You (q to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in {"q", "quit", "exit"}:
                    break
                if not question:
                    continue
            reply = client.ask(question)
            print(f"Assistant: {reply.text}")
            print(f"[pi_rpc] {reply.elapsed_s:.3f}s, tools={reply.tools}, "
                  f"tool_s={reply.tool_seconds}, tokens_in={reply.input_tokens}, "
                  f"tokens_out={reply.output_tokens}", flush=True)
            if args.prompt is not None:
                break


if __name__ == "__main__":
    main()
