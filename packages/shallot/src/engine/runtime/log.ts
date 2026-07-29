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

/**
 * force a logging pipeline's ring buffer to read back, and return the lines it produced.
 *
 * The readback fires only on a typegpu-*owned* dispatch, and shallot owns its own encoders and passes
 * (`.with(encoder)`), so a kernel's logs would otherwise never surface. A zero-workgroup dispatch runs
 * no threads but still triggers the drain. Keep it a deliberate debugging call — never frame wiring: a
 * present-but-unfired log taxes every owned dispatch 3.5–8×, while a log-free kernel pays nothing.
 *
 * The wait is a poll, because there is nothing to await: typegpu maps the ring buffer and prints from a
 * `.then` it never hands back, and that lands well after `onSubmittedWorkDone`. One drained dispatch
 * deserializes its whole ring in a single callback, so the first line landing means the batch has —
 * `timeoutMs` only bounds the case where the kernel logged nothing at all.
 * @example
 * const lines = await drainLog(pipeline); // ["hit count 42"]
 * @internal
 */
export async function drainLog(
    pipeline: { dispatchWorkgroups(x: number): void },
    timeoutMs = 1000,
): Promise<string[]> {
    const log = captured();
    const at = log?.lines.length ?? 0;
    pipeline.dispatchWorkgroups(0);
    if (!log) return [];
    const deadline = performance.now() + timeoutMs;
    while (log.lines.length === at && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 1));
    }
    return log.lines.slice(at);
}
