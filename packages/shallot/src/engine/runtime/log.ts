/**
 * a captured GPU-side log line: what a `console.log` inside a TGSL kernel printed, once its ring buffer
 * was read back. `errors` holds the `console.error` subset, which `shallot verify` fails on.
 */
export interface GpuLog {
    lines: string[];
    errors: string[];
}

// typegpu prefixes every GPU line with `%c<messagePrefix>%c ` and two CSS argument strings, then the
// deserialized values. The prefix is the only thing distinguishing a GPU line from any other page log,
// since both arrive through the same `console` methods.
const MARK = "%c GPU %c ";

// read through a cast rather than a `declare global`: the capture object is a harness-side affordance,
// and declaring it globally would put `__gpuLog` in every consumer's ambient types.
const captured = (): GpuLog | undefined => (globalThis as { __gpuLog?: GpuLog }).__gpuLog;

let patched = false;
let drainTail: Promise<void> = Promise.resolve();
let drainPoison: Error | undefined;

function text(args: unknown[]): string {
    const head = String(args[0]).slice(MARK.length);
    const rest = args.slice(3).map((a) => (typeof a === "string" ? a : JSON.stringify(a)));
    return rest.length > 0 ? `${head} ${rest.join(" ")}` : head;
}

/**
 * tee GPU-side log lines into `globalThis.__gpuLog`, so a harness can read what a kernel printed.
 *
 * Opt-in by presence: the host creates `__gpuLog` before the page's own scripts run (`shallot verify`
 * does it in an init script), and without it this is a no-op — an engine that always wrapped `console`
 * would be taxing every app for a debugging affordance.
 *
 * **Call it before any pipeline resolves.** typegpu captures the `console` method *at shader-generation
 * time* and calls that captured reference at readback, so a patch installed after a logging kernel
 * resolves never sees its lines. {@link requestGPU} calls this first thing, which is early enough for
 * every engine pipeline.
 * @internal
 */
export function captureGpuLog(): void {
    const log = captured();
    if (!log || patched) return;
    patched = true;
    for (const level of ["log", "warn", "error"] as const) {
        const original = console[level];
        console[level] = (...args: unknown[]) => {
            if (typeof args[0] === "string" && args[0].startsWith(MARK)) {
                const line = text(args);
                log.lines.push(line);
                if (level === "error") log.errors.push(line);
            }
            original.apply(console, args as never);
        };
    }
}

async function drain(trigger: () => void, timeoutMs: number): Promise<string[]> {
    const log = captured();
    const at = log?.lines.length ?? 0;
    trigger();
    if (!log) return [];
    const deadline = performance.now() + timeoutMs;
    while (log.lines.length === at && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 1));
    }
    if (log.lines.length === at) {
        drainPoison ??= new Error(
            `drainLog timed out after ${timeoutMs}ms; this capture session is poisoned`,
        );
        throw drainPoison;
    }
    return log.lines.slice(at);
}

/**
 * run one TypeGPU-owned logging trigger and return the GPU-log batch its readback drains.
 *
 * The readback fires only when TypeGPU owns and submits the dispatch or draw. A pipeline invoked through
 * `.with(encoder)` or `.with(pass)` cannot drain. A later TypeGPU-owned replay reads the same ring, so
 * its batch includes both the external-pass backlog and the replay's own lines; this API cannot assign
 * those lines to one draw after the fact. Replay only when that duplication is safe. Otherwise inspect
 * the pass-adjacent resource with `probeBuffer` / `probeTexture`. Keep this a deliberate debugging call,
 * never frame wiring: logging injects atomics and extra bindings.
 *
 * Calls are single-flight. An overlapping call waits to trigger until the prior batch finishes, because
 * TypeGPU exposes one fire-and-forget map callback rather than a per-trigger completion promise. The wait
 * polls the captured console: one readback deserializes its whole ring synchronously, so the first new
 * line means that batch has landed. Lines already printed before this call are excluded; lines queued in
 * the GPU ring by an external pass are necessarily included. Single-flight covers calls through this API,
 * not an uncoordinated TypeGPU-owned draw.
 *
 * A timeout rejects and permanently poisons this page's capture session. Every queued or later call then
 * rejects before its trigger, so a late callback can never be returned as a trustworthy later batch. A
 * fresh page/module instance is the recovery boundary; there is deliberately no in-page reset.
 * @example
 * const lines = await drainLog(() => pipeline.dispatchThreads()); // ["hit count 42"]
 * const fragment = await drainLog(() => pipeline.withColorAttachment(target).draw(3));
 * @internal
 */
export function drainLog(trigger: () => void, timeoutMs = 1000): Promise<string[]> {
    const batch = drainTail.then(() => {
        if (drainPoison) throw drainPoison;
        return drain(trigger, timeoutMs);
    });
    drainTail = batch.then(
        () => undefined,
        () => undefined,
    );
    return batch;
}
