import { Compute, type Mirror, type State } from "@dylanebert/shallot";
import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import { Profile } from "@dylanebert/shallot/extras";
import type {
    HarnessTarget,
    PixelProbe,
    Check as WireCheck,
    Verdict as WireVerdict,
} from "@dylanebert/shallot/harness";
import { assertBudget, isDefaultParams } from "./scenarios/budget-coverage";

// The gym contract — the shared core every scenario depends inward on. A scenario builds a
// deterministic scene, optionally asserts behavioral invariants, and gets the profiler's
// timing for free. Gym is a consumer of the shipped `window.__harness` protocol (`@dylanebert/shallot/
// harness`): `installHarness` below translates a scenario's internal {@link Verdict} to the published
// wire {@link WireVerdict} that `shallot verify` drives. Readback for asserts is `Mirror` (not the legacy
// compute/readback); timing is `window.__benchmark` (ProfilePlugin — GPU timestamps + frame).

/** one behavioral assertion; `detail` is the human-readable value, `data` the machine-readable one.
 *  `data` is an optional flat number map a bench script consumes directly (the physics scenario's
 *  `measured` reporter publishes its per-step spans + health counters here). Translated to the published
 *  {@link WireCheck} (`pass` → `ok`) at the harness boundary. */
export interface Check {
    name: string;
    pass: boolean;
    detail?: string;
    data?: Record<string, number>;
}

/** the gym's internal scenario result — what a scenario's `assert` plus the profiler produce. The metrics
 *  ride through to the wire verdict as a pass-through extra; the checks are translated to {@link WireCheck}. */
export interface Verdict {
    metrics?: BenchmarkMeasurement;
    checks?: Check[];
}

// A scenario's tunables are declared as data — the single source of truth the URL parses and the bench
// `--param` sets. `rebuild: true` marks a structural knob (scene size/shape) rather than one the
// scenario's systems re-read each frame; `default` is the value when the URL omits it.
export type Param =
    | {
          key: string;
          type: "bool";
          default: boolean;
          label?: string;
          rebuild?: boolean;
      }
    | {
          key: string;
          type: "select";
          default: string;
          options: string[];
          label?: string;
          rebuild?: boolean;
      }
    | {
          key: string;
          type: "number";
          default: number;
          min?: number;
          max?: number;
          step?: number;
          label?: string;
          rebuild?: boolean;
      };

/** resolved param values keyed by {@link Param.key} — the object {@link Scenario.build} receives and the
 *  scenario reads each frame. */
export type Params = Record<string, boolean | string | number>;

// One scenario = one file with zero environment awareness: `params` declares its tunables, `build`
// attaches its own camera to `canvas` + reads the resolved params, `assert` is the behavioral gate (a
// Mirror-readback verdict). Timing comes from the profiler (`BenchmarkMeasurement`), not the scenario.
export interface Scenario {
    name: string;
    params?: Param[];
    /** the generic pixel classifier is inapplicable: either this scenario renders no framed scene, or a
     *  narrow parameterized row owns a stronger framebuffer/reference assertion. Forwarded to the
     *  harness's `noRender`, so `shallot verify` reports "opt-out" and the named verdict stays mandatory. */
    noRender?: boolean | ((params: Params) => boolean);
    /** final-compositor color-tag observables forwarded to the published harness's `pixelProbe` — the
     *  rung that catches a scene whose real content never reached the composited frame despite every
     *  upstream (mesh/draw/timing) signal reading green. Static: the same probes apply across every
     *  `rebuild` param value the scenario declares, so a red-proof mode changes what's *drawn*, never
     *  which tags are checked for. */
    pixelProbe?: PixelProbe[];
    build(
        canvas: HTMLCanvasElement,
        params: Params,
    ): Promise<{ state: State; dispose: () => void }>;
    assert?(state: State): Promise<Check[]>;
    /** an optional param-gated extra-check phase run after {@link assert}, on the live (still-running)
     *  scene — for checks that drive the scene rather than read a settled snapshot (a scripted pointer
     *  drag, a visual-presence walk). Returns `[]` when its opts aren't set, so a plain `run()` (the
     *  the gold gate) is unperturbed. Its checks fold into the same wire verdict as `assert`'s. */
    probe?(state: State, opts: Record<string, unknown>): Promise<Check[]>;
}

/** resolve declared params against the URL query — the value the URL gives (parsed by type), else the
 *  declared default. One resolution, so a bench `--param` and a hand-typed URL drive the same scene. */
export function resolveParams(decls: Param[], query: URLSearchParams): Params {
    const out: Params = {};
    for (const p of decls) {
        const raw = query.get(p.key);
        if (raw === null) {
            out[p.key] = p.default;
        } else if (p.type === "bool") {
            out[p.key] = raw !== "0" && raw !== "false";
        } else if (p.type === "number") {
            const n = Number(raw);
            out[p.key] = Number.isFinite(n) ? n : p.default;
        } else {
            out[p.key] = p.options.includes(raw) ? raw : p.default;
        }
    }
    return out;
}

const registry = new Map<string, Scenario>();

export function register(scenario: Scenario): void {
    if (registry.has(scenario.name)) {
        throw new Error(`gym scenario "${scenario.name}" already registered`);
    }
    registry.set(scenario.name, scenario);
}

export function getScenario(name: string): Scenario | undefined {
    return registry.get(name);
}

export function scenarioNames(): string[] {
    return [...registry.keys()];
}

/** resolve a scenario's static or parameterized generic-pixel exception. */
export function resolveNoRender(noRender: Scenario["noRender"], params: Params): boolean {
    return typeof noRender === "function" ? noRender(params) : noRender === true;
}

/** await `n` animation frames — lets the running render loop advance a known amount. */
export function frames(n: number): Promise<void> {
    return new Promise((resolve) => {
        let i = 0;
        const tick = () => (++i >= n ? resolve() : requestAnimationFrame(tick));
        requestAnimationFrame(tick);
    });
}

// Mirror is 1-2 frames stale by design (a staging ring + async map). After mutating state
// that the GPU pack reads (a camera move), wait until a snapshot encoded *after* now lands,
// so a readback reflects the new state. Bounded — a stuck map resolves to the loop cap, by
// which point the pack has long since re-run anyway.
export async function settle(m: Mirror, max = 120): Promise<void> {
    const target = Compute.frame + 2;
    for (let i = 0; i < max; i++) {
        await frames(1);
        if (m.snapshot && m.snapshot.frame >= target) return;
    }
}

// 20-byte DrawIndexedIndirect record: { indexCount, instanceCount, firstIndex, baseVertex, firstInstance }.
const DRAW_ARG_U32S = 5;

/**
 * per-pair `instanceCount` for one view slot, decoded from a {@link Mirror} of a shallot cull
 * producer's slot-major `drawArgs` buffer (`slot * pairCount + pair`). The compacted survivor
 * count a frustum cull writes — the GPU→CPU assert input every culling scenario reads. `null`
 * until the first snapshot resolves.
 */
export function packCounts(m: Mirror, slot: number, pairCount: number): Uint32Array | null {
    if (!m.snapshot) return null;
    const args = new Uint32Array(m.snapshot.bytes);
    const out = new Uint32Array(pairCount);
    for (let p = 0; p < pairCount; p++) {
        out[p] = args[(slot * pairCount + p) * DRAW_ARG_U32S + 1];
    }
    return out;
}

// Install the published `window.__harness` (`@dylanebert/shallot/harness`) that `shallot verify` drives.
// `run` measures through the profiler (timing source of truth), then asserts, then translates the internal
// verdict to the wire shape (`pass` → `ok`, metrics ride through as a pass-through extra). `ready` waits on
// both the built scene and the profiler's first resolved frame. `opts` carries the CLI's `--query` params
// as strings (the URL is the primary channel); warmup/frames coerce to the profiler defaults when absent.
export function installHarness(
    scenario: Scenario,
    state: State,
    built: () => boolean,
    params: Params,
): void {
    const target: HarnessTarget = {
        ...(resolveNoRender(scenario.noRender, params) ? { noRender: true } : {}),
        ...(scenario.pixelProbe?.length ? { pixelProbe: scenario.pixelProbe } : {}),
        get ready() {
            return built() && window.__benchmark?.ready === true;
        },
        async run(opts?: Record<string, unknown>): Promise<WireVerdict> {
            const benchmark = window.__benchmark;
            if (!benchmark) {
                throw new Error("ProfilePlugin missing — window.__benchmark not installed");
            }
            // `??` not `||` — warmup=0 (measure from frame zero) is a legitimate value.
            const warmup = opts?.warmup != null ? Number(opts.warmup) : 60;
            const frames = opts?.frames != null ? Number(opts.frames) : 500;
            const metrics: BenchmarkMeasurement = await benchmark.measure(warmup, frames);
            const asserted = await scenario.assert?.(state);
            // the probe drives the live scene (pointer drag, visual walk); it returns [] unless its opts
            // are set, so the standard gold gate stays a pure assert. Its checks join assert's in one verdict.
            const probed = await scenario.probe?.(state, opts ?? {});
            // the compile/memory structural budget: folded into every
            // scenario's verdict generically, off `SCENARIO_BUDGETS`/`BUDGET_EXEMPTIONS` data alone — a
            // scenario earns the check by having a budget entry, never by editing its own `assert`. Reads
            // `Profile` directly (not `metrics.compile`, which isn't filtered to real pipelines).
            // `gpuBytes` excludes `Profile.lazyBytes` — bytes from an allocation the allocator marked
            // lazily-grown (a pool that grows under real GPU backpressure, `LazyAlloc`) are not exact for
            // a fixed scenario at fixed params, so the gate reads the total minus them, the quantity that
            // IS exact by mechanism.
            const budgeted = assertBudget(
                scenario.name,
                isDefaultParams(scenario.params ?? [], params),
                {
                    pipelines: [...Profile.compile.keys()].filter((k) =>
                        Profile.compiledPipelines.has(k),
                    ).length,
                    gpuBytes: Profile.bufferBytes + Profile.textureBytes - Profile.lazyBytes,
                    // the label-discipline instrument, not an axis: raw constructor calls, which the
                    // filtered count above can only equal while every descriptor label is per-pipeline
                    // (`budget:labels`, `budget-coverage.ts`).
                    pipelineCalls: Profile.pipelineCalls,
                },
            );
            const checks =
                asserted || probed || budgeted.length > 0
                    ? [...(asserted ?? []), ...(probed ?? []), ...budgeted]
                    : undefined;
            const wire: WireCheck[] | undefined = checks?.map((c) => ({
                name: c.name,
                ok: c.pass,
                detail: c.detail,
                data: c.data,
            }));
            return {
                ok: wire ? wire.every((c) => c.ok) : true,
                ...(wire ? { checks: wire } : {}),
                metrics,
            };
        },
    };
    window.__harness = target;
}
