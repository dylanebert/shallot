import type { State } from "../engine";
import type { PixelProbe } from "./pixels";

export { type PixelProbe, type PixelProbeResult, pixelProbePass, probePixels } from "./pixels";

// The published verification protocol. A project installs `window.__harness` so a driver can run it
// in a real browser: wait for `ready`, call `run(opts)`, read the `Verdict`, exit 0/nonzero. This
// module runs IN THE PAGE — it never imports Playwright or node. The driver runs in node/bun and never
// imports this. The two meet only over the `window.__harness` shape and the JSON `Verdict` on the wire.

/**
 * one named check inside a {@link Verdict}: a boolean claim the project asserted, with optional
 * human detail on a failure.
 */
export interface Check {
    name: string;
    ok: boolean;
    detail?: string;
    /** machine-readable diagnostics behind the check — the counterpart to the human `detail`. A
     *  benchmark or regression atom fills it with structured numbers (per-step spans, entity counts) a
     *  driving script reads back from the `--json` verdict; ignored by the pass/fail decision. */
    data?: Record<string, number>;
}

/**
 * the result the driver reads back from {@link HarnessTarget.run}. `ok` is the pass/fail the
 * command's exit code follows; `checks` are the named assertions behind it. Extra fields pass
 * through verbatim into the command's `--json` output, so a project can report richer diagnostics
 * (frame times, entity counts) without a protocol change.
 *
 * @example
 * ```
 * const harness = installHarness(app.state);
 * harness.run = async () => ({ ok: true, checks: [{ name: "scene booted", ok: true }] });
 * ```
 */
export interface Verdict {
    ok: boolean;
    checks?: Check[];
    [extra: string]: unknown;
}

/**
 * the contract a project installs on `window.__harness` for the driver to drive. `ready`
 * gates the run (the command waits for it before calling `run`); `run` returns the pass/fail
 * {@link Verdict}. {@link installHarness} installs a default.
 */
export interface HarnessTarget {
    /** true once the scene has built and drawn at least one frame — the command waits for this. */
    ready: boolean;
    /** declare that this target renders no framed scene by design — a GPU-compute-only microbench, a
     *  solid-fill clip test. The driver's pixel gate then reports `rendered: "opt-out"` and passes
     *  on the verdict alone, rather than failing the blank canvas. Omit (or `false`) for any target that
     *  draws content — the pixel gate is the honesty check that a green verdict didn't ride over a canvas
     *  that silently rendered nothing. The opt-out is visible in the run output, never a silent exemption. */
    noRender?: boolean;
    /** one or more final-compositor color-tag observables the driver checks against the post-run
     *  compositor screenshot, appended to the {@link Verdict}'s `checks` and folded into the command's
     *  pass/fail. Upstream evidence (mesh/draw counts, timings, error absence) proves only its own rung —
     *  a {@link PixelProbe} is the rung that catches a scene whose real content never reached the composited
     *  frame despite every upstream signal reading green. Opt-in: a
     *  harness that renders no framed scene declares `noRender` instead, never both. */
    pixelProbe?: PixelProbe[];
    /** run the verification and resolve a {@link Verdict}. `opts` carries the command's `--query`
     *  values (URL params are the primary channel; this mirrors them for programmatic runs). */
    run?(opts?: Record<string, unknown>): Promise<Verdict>;
}

// the `window.__harness` slot the driver reads — the one global the published protocol names, so a
// project can install its target with a plain `window.__harness = target` and have it typed.
declare global {
    interface Window {
        __harness?: HarnessTarget;
    }
}

/**
 * install the default `window.__harness` for the driver and return the handle. `ready` flips true
 * once `state.time.elapsed` advances past zero (the first step), and `run` reports a booted pass (the
 * driver's pixel gate derives the real `rendered` verdict; the default run only attests the scene booted).
 * A project layers its own assertions by replacing `run` on the returned handle:
 *
 * @example
 * ```
 * const app = await run({ scene });
 * const harness = installHarness(app.state);
 * harness.run = async () => ({ ok: true, checks: [{ name: "scene booted", ok: true }] });
 * ```
 */
export function installHarness(state: State): HarnessTarget {
    const target: HarnessTarget = {
        // elapsed advances only after the first frame steps, so `ready` marks the built scene's first draw.
        get ready(): boolean {
            return state.time.elapsed > 0;
        },
        run: async (): Promise<Verdict> => ({
            ok: true,
            checks: [{ name: "booted", ok: true }],
        }),
    };
    // assign via globalThis, not `window`: the module runs in the page, but its unit tests run under bun
    // where `window` doesn't exist — globalThis is the one name defined in both.
    (globalThis as unknown as Window).__harness = target;
    return target;
}
