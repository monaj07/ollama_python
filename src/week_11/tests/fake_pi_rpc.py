"""Tiny Pi RPC protocol fixture: no Pi, Ollama, camera, microphone or SQLite."""
import json
import sys
import time

count = 0
for line in sys.stdin:
    cmd = json.loads(line)
    kind = cmd.get("type")
    rid = cmd.get("id")
    if kind == "get_state":
        print(json.dumps({"type": "response", "id": rid, "command": "get_state", "success": True, "data": {"sessionId": "mock"}}), flush=True)
        continue
    if kind != "prompt":
        continue
    count += 1
    question = cmd["message"]
    if question == "hang":
        time.sleep(3)
        continue
    if question == "reject":
        print(json.dumps({"type": "response", "id": rid, "command": "prompt", "success": False, "error": "mock rejection"}), flush=True)
        continue
    print(json.dumps({"type": "response", "id": rid, "command": "prompt", "success": True, "data": {"disposition": "started"}}), flush=True)
    print(json.dumps({"type": "agent_start"}), flush=True)
    if question == "tool":
        print(json.dumps({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "query_memory", "args": {"operation": "summary"}}), flush=True)
        print(json.dumps({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "query_memory", "isError": False, "durationMs": 250}), flush=True)
        # A model's earlier tool-call response must not become spoken output.
        print(json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse", "content": [{"type": "toolCall", "name": "query_memory"}], "usage": {"input": 6, "output": 3}}}), flush=True)
    final_reason = "length" if question == "truncated" else "stop"
    msg = {"role": "assistant", "stopReason": final_reason, "content": [
        {"type": "thinking", "thinking": "Don't speak this."},
        {"type": "text", "text": "answer " + str(count)}], "usage": {"input": 8, "output": 2}}
    print(json.dumps({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "partial answer"}}), flush=True)
    print(json.dumps({"type": "message_end", "message": msg}), flush=True)
    print(json.dumps({"type": "agent_end", "messages": [msg], "willRetry": False}), flush=True)
    print(json.dumps({"type": "agent_settled", "aborted": False}), flush=True)
