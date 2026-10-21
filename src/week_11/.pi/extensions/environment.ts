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
  let sessionId = "";
  let needsTurn = true;
  let beginPromise: Promise<void> | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = ctx.sessionManager.getSessionId();
    needsTurn = true;
    beginPromise = undefined;
  });
  pi.on("agent_start", () => {
    needsTurn = true;
    beginPromise = undefined;
  });
  pi.on("session_shutdown", () => bridge.close());

  // Lazy: general chat never starts the Python worker. A shared promise keeps
  // simultaneous tool calls from resetting the previous PIR reading mid-turn.
  async function ensureTurn(): Promise<void> {
    if (!needsTurn) {
      if (beginPromise) await beginPromise;
      return;
    }
    needsTurn = false;
    beginPromise = (async () => {
      const reply = await bridge.request("begin_turn", { session_id: sessionId }, 5000);
      if (!reply.ok) throw new Error(reply.error || "Cannot start sensor turn");
    })();
    try {
      await beginPromise;
    } catch (err) {
      needsTurn = true;
      throw err;
    }
  }
  async function sensorRequest(tool: string, args: unknown, timeoutMs: number, signal?: AbortSignal) {
    await ensureTurn();
    return bridge.request(tool, args, timeoutMs, signal);
  }
  // Keep full structured results for inspection while sending a compact model view.
  function result(reply: Reply, modelView?: Reply) {
    return { content: [{ type: "text" as const, text: JSON.stringify(modelView ?? reply) }], details: reply };
  }

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
      const reply = await sensorRequest("detect_motion", params, 25_000, signal);
      if (!reply.ok) throw new Error(reply.error || "PIR tool failed");
      return result(reply);
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
      const reply = await sensorRequest("observe_scene", params, 190_000, signal);
      if (!reply.ok) throw new Error(reply.error || "Camera tool failed");
      return result(reply);
    },
  });

  // Pi performs deterministic nested orchestration. Each child tool goes through
  // Pi's normal validation, hooks and tracing, but requires only one LLM decision.
  pi.registerTool({
    name: "check_presence",
    label: "Check room presence",
    description: "For live room-presence questions, ALWAYS check PIR and camera, even if PIR reports no motion. The result contains an evidence_summary composed deterministically from both sensor readings: use it faithfully. Camera detections do not establish verified occupancy.",
    parameters: Type.Object({}, { additionalProperties: false }),
    async execute(_id, _params, signal, _onUpdate, ctx) {
      const started = performance.now();
      async function runNested(name: string, args: object): Promise<Reply> {
        try {
          const outcome = await ctx.executeTool(name, args, { signal });
          const reply = outcome.result.details as Reply | undefined;
          if (outcome.isError || !reply || typeof reply.ok !== "boolean") {
            return { ok: false, error: `Nested ${name} failed` };
          }
          return reply;
        } catch (err) {
          return { ok: false, error: `${name}: ${String(err)}`.slice(0, 400) };
        }
      }
      const pir = await runNested("detect_motion", {});
      // Deliberately do not skip the camera if the PIR is negative or unavailable.
      const camera = signal?.aborted
        ? { ok: false, error: "Cancelled before camera capture" }
        : await runNested("observe_scene", { targets: ["person"], conf: 0.2 });
      const rawCount = camera.ok && camera.counts && typeof camera.counts === "object"
        ? (camera.counts as Record<string, unknown>).person : null;
      const count = typeof rawCount === "number" && Number.isInteger(rawCount) && rawCount >= 0
        ? rawCount : null;
      const motion = pir.ok && typeof pir.motion_signal_active === "boolean"
        ? pir.motion_signal_active : null;
      const observations = Array.isArray(camera.detections) ? camera.detections : [];
      const scores = observations.filter((d) => d && typeof d === "object"
        && d.label === "person" && typeof d.confidence === "number"
        && Number.isFinite(d.confidence) && d.confidence >= 0 && d.confidence <= 1)
        .map((d) => d.confidence as number);
      // A deterministic sentence helps the small model avoid reversing a boolean
      // sensor signal (e.g. claiming movement when PIR reported false).
      const pirEvidence = motion === true ? "PIR: movement detected."
        : motion === false ? "PIR: no movement detected."
        : "PIR: reading unavailable.";
      const cameraEvidence = count === null ? "Camera: reading unavailable."
        : count === 0 ? "Camera: no person detected."
        : `Camera: ${count} tentative person detection${count === 1 ? "" : "s"}.`;
      const combined = {
        ok: pir.ok === true && camera.ok === true,
        partial: !(pir.ok === true && camera.ok === true),
        motion_signal_active: motion,
        camera_person_count: count,
        evidence_summary: `${pirEvidence} ${cameraEvidence} These readings do not establish verified occupancy.`,
        max_person_score: count !== null && count > 0 && scores.length ? Math.max(...scores) : null,
        pir_observed_at: pir.observed_at ?? null,
        camera_observed_at: camera.observed_at ?? null,
        evidence_path: camera.source ?? null,
        recording: camera.recording ?? null,
        elapsed_s: Math.round((performance.now() - started) / 10) / 100,
        errors: {
          pir: pir.ok ? null : pir.error ?? "No valid PIR reading",
          camera: camera.ok ? null : camera.error ?? "No valid camera reading",
        },
        limitations: "PIR indicates movement, not occupancy. Camera counts are tentative single-frame detections, not verified people; zero does not prove absence. Detector scores are not calibrated probabilities.",
      };
      return result(combined);
    },
  });

  // One Pi-facing memory tool. Internal operations remain validated and read-only.
  // Small local models choose a single operation instead of searching many tools.
  pi.registerTool({
    name: "query_memory",
    label: "Query saved environmental memory",
    description: "READ-ONLY SQLite memory; never activates sensors. Choose operation: recent (latest 1-10 observations), summary (counts across all saved observations), away (latest Away Mode report), events (list up to 12 person-positive detections with timestamps and scores), event (one specific event ID with evidence metadata). For follow-ups preserve the current scope; an Away Mode follow-up stays about Away Mode. This tool has no SQL, delete or capture operations.",
    parameters: Type.Object({
      operation: Type.Union([
        Type.Literal("recent"), Type.Literal("summary"), Type.Literal("away"),
        Type.Literal("events"), Type.Literal("event"),
      ]),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 12,
        description: "recent: 1-10 (default 1); events: 1-12 (default 12). Not used otherwise." })),
      event_id: Type.Optional(Type.Integer({ minimum: 1, maximum: 2147483647,
        description: "Required only for operation=event. Exact saved observation ID." })),
    }, { additionalProperties: false }),
    async execute(_id, params, signal) {
      const reply = await bridge.request("query_memory", params, 10_000, signal);
      return result(reply);
    },
  });

}
