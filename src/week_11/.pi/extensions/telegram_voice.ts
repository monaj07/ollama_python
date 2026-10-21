/** Convert inbound Telegram voice attachments to text before Pi calls the LLM.
 * Reuses Week 09 whisper.cpp. No model-callable STT tool is registered.
 * Works only with voice files downloaded by the Telegram extension into cwd.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { realpath, stat } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const week11 = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const python = resolve(week11, "../week_09/.venv/bin/python");
const script = resolve(week11, "telegram_stt.py");
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_STDOUT = 8192;
const TIMEOUT_MS = 100000;

function telegramVoicePath(input: string): string | null {
  if (!input.includes("[telegram attachment]")) return null;
  const matches = [...input.matchAll(/^\s*-\s*voice:\s*.+?=>\s*(\/[^\r\n]+-voice\.oga)\s*$/gm)];
  if (matches.length !== 1) return null;
  return matches[0][1];
}

async function ensureValidFile(inputPath: string, cwd: string): Promise<string> {
  const root = await realpath(cwd);
  const candidate = await realpath(inputPath);
  const rel = relative(root, candidate);
  if (!rel || rel.startsWith("..") || isAbsolute(rel)) throw new Error("Voice file is outside project");
  if (!candidate.endsWith("-voice.oga")) throw new Error("Invalid voice attachment suffix");
  const info = await stat(candidate);
  if (!info.isFile() || info.size === 0 || info.size > MAX_BYTES) throw new Error("Invalid voice attachment size");
  return candidate;
}

function transcribe(filePath: string, cwd: string): Promise<string> {
  return new Promise((ok, fail) => {
    const child = spawn(python, [script, "--input", filePath, "--allowed-root", cwd], {
      cwd, stdio: ["ignore", "pipe", "pipe"],
    });
    let out = "";
    let err = "";
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill("SIGKILL"); }, TIMEOUT_MS);
    child.stdout.on("data", (chunk: Buffer) => {
      out += chunk.toString("utf8");
      if (out.length > MAX_STDOUT) child.kill("SIGKILL");
    });
    child.stderr.on("data", (chunk: Buffer) => { err = (err + chunk.toString("utf8")).slice(-2048); });
    child.on("error", (error) => { clearTimeout(timer); fail(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) return fail(new Error("Whisper timed out"));
      if (code !== 0) return fail(new Error(`Whisper exited ${code}: ${err.slice(-300)}`));
      if (!out.trim()) return fail(new Error("Whisper returned no speech"));
      ok(out.trim());
    });
  });
}

export default function (pi: ExtensionAPI) {
  pi.on("input", async (event, ctx) => {
    // pi-telegram-plus delivers downloaded voice attachments as source="interactive".
    // Do not gate on event.source: match the attachment shape and validate its file.
    const rawPath = telegramVoicePath(event.text);
    if (!rawPath) return { action: "continue" };
    try {
      const path = await ensureValidFile(rawPath, ctx.cwd);
      const transcript = await transcribe(path, ctx.cwd);
      // Preserve any accompanying caption but do not send raw attachment tokens to the LLM.
      const caption = event.text.split("[telegram attachment]")[0].trim();
      return { action: "transform", text: caption ? `${caption}\n\n${transcript}` : transcript };
    } catch (error) {
      // Fail closed: don't send an attachment path or file contents to the model.
      ctx.ui.notify("Telegram voice transcription failed; please resend as text.", "warning");
      return { action: "transform", text: "The Telegram voice message could not be transcribed. Please ask for a text message instead." };
    }
  });
}
