import type { State } from "../engine";
import { Physics } from "../standard/physics";
import { Transform } from "../standard/transforms";

// The published verification protocol. A project installs `window.__harness` so `shallot verify`
// (bin/verify.ts) can drive it in a real browser: wait for `ready`, call `run(opts)`, read the
// `Verdict`, exit 0/nonzero. This module runs IN THE PAGE — it never imports Playwright or node.
// The `verify` command runs in node/bun and never imports this. The two meet only over the
// `window.__harness` shape and the JSON `Verdict` on the wire.

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
 * the result `shallot verify` reads back from {@link HarnessTarget.run}. `ok` is the pass/fail the
 * command's exit code follows; `checks` are the named assertions behind it. Extra fields pass
 * through verbatim into the command's `--json` output, so a project can report richer diagnostics
 * (frame times, entity counts) without a protocol change.
 *
 * @example
 * ```
 * const harness = installHarness(app.state);
 * harness.run = async () => {
 *     const fell = harness.read!(boxEid)!.pos[1] < 0.5;
 *     return { ok: fell, checks: [{ name: "box fell", ok: fell }] };
 * };
 * ```
 */
export interface Verdict {
    ok: boolean;
    checks?: Check[];
    [extra: string]: unknown;
}

/**
 * one entity's live pose, as {@link HarnessTarget.read} returns it: the physics-owned pose for a
 * `Body`, else the authored `Transform`. `vel` is present only for a physics body.
 */
export interface PoseState {
    pos: [number, number, number];
    quat: [number, number, number, number];
    vel?: [number, number, number];
}

/**
 * the contract a project installs on `window.__harness` for `shallot verify` to drive. `ready`
 * gates the run (the command waits for it before calling `run`); `run` returns the pass/fail
 * {@link Verdict}; `read` exposes a live entity pose so an assertion can check where something
 * ended up without the project hand-rolling a readback. {@link installHarness} installs a default.
 */
export interface HarnessTarget {
    /** true once the scene has built and drawn at least one frame — the command waits for this. */
    ready: boolean;
    /** declare that this target renders no framed scene by design — a GPU-compute-only microbench, a
     *  solid-fill clip test. `shallot verify`'s pixel gate then reports `rendered: "opt-out"` and passes
     *  on the verdict alone, rather than failing the blank canvas. Omit (or `false`) for any target that
     *  draws content — the pixel gate is the honesty check that a green verdict didn't ride over a canvas
     *  that silently rendered nothing. The opt-out is visible in the run output, never a silent exemption. */
    noRender?: boolean;
    /** run the verification and resolve a {@link Verdict}. `opts` carries the command's `--query`
     *  values (URL params are the primary channel; this mirrors them for programmatic runs). */
    run?(opts?: Record<string, unknown>): Promise<Verdict>;
    /** the live pose of an entity by eid — physics pose for a `Body`, else its `Transform`.
     *  `null` when the eid carries neither. */
    read?(eid: number): PoseState | null;
}

// the `window.__harness` slot `shallot verify` reads — the one global the published protocol names, so a
// project (or the gym) can install its target with a plain `window.__harness = target` and have it typed.
declare global {
    interface Window {
        __harness?: HarnessTarget;
    }
}

/**
 * install the default `window.__harness` for `shallot verify` and return the handle. `ready` flips true
 * once a frame has drawn, `read` returns live entity poses, and `run` reports a booted pass (verify's
 * pixel gate derives the real `rendered` verdict; the default run only attests the scene booted).
 * A project layers its own assertions by replacing `run` on the returned handle:
 *
 * @example
 * ```
 * const app = await run({ scene });
 * const harness = installHarness(app.state);
 * harness.run = async () => {
 *     const box = harness.read!(boxEid)!;
 *     return { ok: box.pos[1] < 0.5, checks: [{ name: "box settled", ok: box.pos[1] < 0.5 }] };
 * };
 * ```
 */
export function installHarness(state: State): HarnessTarget {
    const target: HarnessTarget = {
        // elapsed advances only after the first frame steps — so a truthy read means build finished
        // (this ran) and the RAF loop has driven at least one draw.
        get ready(): boolean {
            return state.time.elapsed > 0;
        },
        read(eid: number): PoseState | null {
            const body = Physics.backend?.readBody(eid);
            if (body) {
                return {
                    pos: [body.pos[0], body.pos[1], body.pos[2]],
                    quat: [body.quat[0], body.quat[1], body.quat[2], body.quat[3]],
                    vel: [body.vel[0], body.vel[1], body.vel[2]],
                };
            }
            if (state.has(eid, Transform)) {
                const p = Transform.pos.read(eid, new Float32Array(4));
                const r = Transform.rot.read(eid, new Float32Array(4));
                return { pos: [p[0], p[1], p[2]], quat: [r[0], r[1], r[2], r[3]] };
            }
            return null;
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
