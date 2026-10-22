/** Pi-local Bluetooth speech for settled replies (including Telegram-origin turns). Never sends audio to Telegram. */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { resolve } from "node:path";

export default function (pi: ExtensionAPI) {
  let finalText = "";

  pi.on("before_agent_start", () => { finalText = ""; });

  // message_end gives the completed response; agent_settled waits for tools/retries.
  pi.on("message_end", (event) => {
    const message = event.message;
    if (message.role !== "assistant") return;
    // Ignore an assistant turn that is still calling tools, or an error/abort.
    if (message.stopReason !== "stop") return;
    finalText = message.content
      .filter((block) => block.type === "text")
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("\n")
      .trim();
  });

  pi.on("agent_settled", async (event, ctx) => {
    // Speak in the local Pi TUI (including Telegram-controlled turns), never RPC.
    if (ctx.mode !== "tui" || event.aborted || !finalText) {
      finalText = "";
      return;
    }
    const text = finalText;
    finalText = ""; // prevent duplicate speech if settle fires again
    const script = resolve(ctx.cwd, "local_speaker.py");
    const python = resolve(ctx.cwd, "../week_09/.venv/bin/python");
    await new Promise<void>((finish) => {
      const child = spawn(python, [script], { cwd: ctx.cwd, stdio: ["pipe", "ignore", "pipe"] });
      let errors = "";
      child.stdin.on("error", () => {}); // child may exit before stdin closes
      child.stderr.on("data", (data: Buffer) => {
        errors = (errors + data.toString()).slice(-500);
      });
      child.on("error", (error) => {
        ctx.ui.notify(`[local-speaker] ${error.message}`, "warning");
        finish();
      });
      child.on("close", (code) => {
        if (code !== 0) ctx.ui.notify(`[local-speaker] ${errors || `exit ${code}`}`, "warning");
        finish();
      });
      child.stdin.end(text);
    });
  });
}
