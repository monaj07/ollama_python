/** Native Pi tools backed by one persistent Python process per Pi session. */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const extensionDir = dirname(fileURLToPath(import.meta.url));
const workerPath = resolve(extensionDir, "../../physical_worker.py");
const repoRoot = resolve(extensionDir, "../../../..");
const python = process.env.PI_WEEK11_PYTHON || "python3";

type Reply = Record<string, unknown> & { ok: boolean; error?: string };
type Pending = {
  resolve: (reply: Reply) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  cleanup: () => void;
};

class PhysicalBridge {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending: Pending | null = null;
  private buffer = "";
  private queue: Promise<void> = Promise.resolve();

  private fail(error: Error) {
    const p = this.pending;
    this.pending = null;
    if (p) {
      clearTimeout(p.timer);
      p.cleanup();
      p.reject(error);
    }
    const old = this.child;
    this.child = null;
    this.buffer = "";
    if (old && !old.killed) old.kill("SIGTERM");
  }

  private ensureWorker(): ChildProcessWithoutNullStreams {
    if (this.child && this.child.exitCode === null && !this.child.killed) return this.child;
    this.buffer = "";
    // stdout is a machine-readable protocol; stderr remains visible for debugging.
    const child = spawn(python, ["-u", workerPath], {
      cwd: repoRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(chunk); // drain stderr to avoid subprocess deadlock
    });
    child.stdout.on("data", (chunk: string) => {
      if (this.child !== child) return;
      this.buffer += chunk;
      if (this.buffer.length > 1_048_576) {
        this.fail(new Error("Python bridge response exceeds 1 MB"));
        return;
      }
      let newline: number;
      while ((newline = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, newline);
        this.buffer = this.buffer.slice(newline + 1);
        if (!line.trim()) continue;
        const p = this.pending;
        this.pending = null;
        if (!p) {
          this.fail(new Error("Unsolicited Python bridge output"));
          return;
        }
        clearTimeout(p.timer);
        p.cleanup();
        try {
          const reply = JSON.parse(line) as Reply;
          if (typeof reply !== "object" || reply === null || typeof reply.ok !== "boolean") {
            throw new Error("Invalid JSON reply shape");
          }
          p.resolve(reply);
        } catch {
          p.reject(new Error("Python bridge sent invalid JSON"));
          this.fail(new Error("Python bridge protocol failed"));
        }
      }
    });
    child.on("error", (err) => {
      if (this.child === child) this.fail(new Error(`Cannot start Python bridge: ${err.message}`));
    });
    child.on("exit", (code) => {
      if (this.child === child) this.fail(new Error(`Python bridge exited (code ${code})`));
    });
    return child;
  }

  request(tool: string, args: unknown, timeoutMs: number, signal?: AbortSignal): Promise<Reply> {
    const job = this.queue.then(() => new Promise<Reply>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new Error("Tool call cancelled"));
        return;
      }
      const child = this.ensureWorker();
      const onAbort = () => this.fail(new Error("Tool call cancelled"));
      signal?.addEventListener("abort", onAbort, { once: true });
      const timer = setTimeout(() => this.fail(new Error(`Tool timed out after ${timeoutMs} ms`)), timeoutMs);
      this.pending = {
        resolve, reject, timer,
        cleanup: () => signal?.removeEventListener("abort", onAbort),
      };
      child.stdin.write(JSON.stringify({ tool, args }) + "\n", (err) => {
        if (err && this.child === child) this.fail(new Error(`Python bridge write failed: ${err.message}`));
      });
    }));
    this.queue = job.then(() => undefined, () => undefined);
    return job;
  }

  close() {
    const child = this.child;
    this.child = null;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.cleanup();
      this.pending.reject(new Error("Pi session ended"));
      this.pending = null;
    }
    if (child) {
      child.stdin.end(); // EOF triggers the Python finally block -> closes YOLOE worker.
      setTimeout(() => {
        if (child.exitCode === null && !child.killed) child.kill("SIGTERM");
      }, 2500).unref();
    }
  }
}

export default function (pi: ExtensionAPI) {
  const bridge = new PhysicalBridge();
  pi.on("session_shutdown", () => bridge.close());

  pi.registerCommand("sensor-health", {
    description: "Check the Python sensor bridge without using an LLM",
    handler: async (_args, ctx) => {
      try {
        const reply = await bridge.request("health", {}, 5000);
        ctx.ui.notify(reply.ok ? "Python sensor bridge: ready" : `Sensor bridge error: ${reply.error}`, reply.ok ? "info" : "error");
      } catch (err) {
        ctx.ui.notify(`Sensor bridge error: ${String(err)}`, "error");
      }
    },
  });

  pi.registerTool({
    name: "detect_motion",
    label: "Read PIR sensor",
    description: "Read the current PIR motion sensor state. Motion is not proof of a person or occupancy.",
    parameters: Type.Object({}, { additionalProperties: false }),
    async execute(_id, params, signal) {
      const reply = await bridge.request("detect_motion", params, 25_000, signal);
      if (!reply.ok) throw new Error(reply.error || "PIR tool failed");
      return { content: [{ type: "text", text: JSON.stringify(reply) }], details: reply };
    },
  });

  pi.registerTool({
    name: "observe_scene",
    label: "Capture and detect",
    description: "Take a fresh camera image and run YOLOE for the requested 1-8 labels. Results are tentative detections, not verified occupancy. For a broad scene use likely labels such as person, cup, bottle, chair, laptop, glasses, plant.",
    parameters: Type.Object({
      targets: Type.Array(Type.String({ minLength: 1, maxLength: 64 }), { minItems: 1, maxItems: 8 }),
      conf: Type.Optional(Type.Number({ minimum: 0.1, maximum: 0.8, description: "Default 0.2" })),
    }, { additionalProperties: false }),
    async execute(_id, params, signal) {
      const reply = await bridge.request("observe_scene", params, 190_000, signal);
      if (!reply.ok) throw new Error(reply.error || "Camera tool failed");
      return { content: [{ type: "text", text: JSON.stringify(reply) }], details: reply };
    },
  });
}
