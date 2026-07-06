import { Compute, type Mirror, type State } from "@dylanebert/shallot";
import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import type { HarnessTarget, RunOpts } from "../../../harness/core";
import type { Check, Verdict } from "../../../harness/gym/verdict";

// The gym contract — the shared core every scenario depends inward on. A scenario builds a
// deterministic scene, optionally asserts behavioral invariants, and gets the profiler's
// timing for free. The launcher (harness/gym) routes the Verdict; this layer never touches
// the transport. Readback for asserts is `Mirror` (not the legacy compute/readback); timing
// is `window.__benchmark` (ProfilePlugin — the source of truth, GPU timestamps + frame).
export type { Check, Verdict };

// A scenario's tunables are declared as data — the single source of truth the URL parses, the bench
// `--param` sets, and the live control panel auto-renders from. A `bool` is a checkbox, `select` a
// dropdown, `number` a numeric input. `rebuild: true` marks a structural knob (scene size/shape):
// changing it reloads the page so the scene rebuilds clean; a live knob (a viz toggle, a mode) mutates
// in place and the scenario's systems read it next frame. `default` is the value when the URL omits it.
// `when` gates a knob's control on the other params — a mode-specific knob declares `when: (v) => v.mode
// === "cull"` so the panel shows only what the current scene actually reads; absent = always shown.
export type Param =
    | {
          key: string;
          type: "bool";
          default: boolean;
          label?: string;
          rebuild?: boolean;
          when?: When;
      }
    | {
          key: string;
          type: "select";
          default: string;
          options: string[];
          label?: string;
          rebuild?: boolean;
          when?: When;
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
          when?: When;
      };

/** relevance predicate over the resolved {@link Params} — `true` shows the control, `false` hides it. */
type When = (values: Params) => boolean;

/** resolved param values keyed by {@link Param.key} — the object {@link Scenario.build} receives and the
 *  scenario reads each frame. Live controls mutate it in place. */
export type Params = Record<string, boolean | string | number>;

// One scenario = one file with zero environment awareness: `params` declares its tunables, `build`
// attaches its own camera to `canvas` + reads the resolved params, `assert` is the behavioral gate (a
// Mirror-readback verdict), `live` is a HUD line for eyeballing it. Headless (the launcher calls `run`)
// and a live tab load the identical page; the scenario never branches on which. Timing comes from the
// profiler (`BenchmarkMeasurement`), not the scenario.
export interface Scenario {
    name: string;
    params?: Param[];
    build(
        canvas: HTMLCanvasElement,
        params: Params,
    ): Promise<{ state: State; dispose: () => void }>;
    assert?(state: State): Promise<Check[]>;
    live?(state: State): string;
}

/** resolve declared params against the URL query — the value the URL gives (parsed by type), else the
 *  declared default. Identical resolution headless and live, so a bench `--param` and a live control
 *  change drive the same scene. */
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

/**
 * render the live control panel from the param declarations — one row per *relevant* param ({@link
 * Param.when}), seeded from `values`. A change mutates `values` in place then calls `onChange(key,
 * rebuild)`: a live param takes effect next frame, a `rebuild` param reloads. When a live change shifts
 * which params are relevant (a mode toggle gating its knobs), the panel re-renders in place. The single
 * source is {@link Scenario.params}; nothing here is scenario-specific. Returns a cleanup that removes it.
 */
export function mountControls(
    host: HTMLElement,
    decls: Param[],
    values: Params,
    onChange: (key: string, rebuild: boolean) => void,
): () => void {
    if (decls.length === 0) return () => {};
    const panel = document.createElement("div");
    panel.className = "controls";

    const relevant = () => decls.filter((p) => p.when?.(values) ?? true);
    let shown = "";

    // a live change can flip which knobs the current scene reads (a mode select gating its group); re-render
    // when the relevant set changes. A `rebuild` change reloads instead, so this only fires for live knobs.
    const sync = () => {
        if (
            relevant()
                .map((p) => p.key)
                .join(",") !== shown
        )
            render();
    };

    const render = () => {
        const vis = relevant();
        shown = vis.map((p) => p.key).join(",");
        panel.replaceChildren();
        for (const p of vis) {
            const row = document.createElement("label");
            row.className = "control";
            const name = document.createElement("span");
            name.textContent = p.label ?? p.key;
            let input: HTMLInputElement | HTMLSelectElement;
            if (p.type === "bool") {
                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.checked = values[p.key] as boolean;
                cb.onchange = () => {
                    values[p.key] = cb.checked;
                    onChange(p.key, p.rebuild === true);
                    sync();
                };
                input = cb;
            } else if (p.type === "select") {
                const sel = document.createElement("select");
                for (const opt of p.options) {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.textContent = opt;
                    sel.appendChild(o);
                }
                sel.value = values[p.key] as string;
                sel.onchange = () => {
                    values[p.key] = sel.value;
                    onChange(p.key, p.rebuild === true);
                    sync();
                };
                input = sel;
            } else {
                const num = document.createElement("input");
                num.type = "number";
                if (p.min !== undefined) num.min = String(p.min);
                if (p.max !== undefined) num.max = String(p.max);
                if (p.step !== undefined) num.step = String(p.step);
                num.value = String(values[p.key]);
                num.onchange = () => {
                    const n = Number(num.value);
                    if (!Number.isFinite(n)) return;
                    values[p.key] = n;
                    onChange(p.key, p.rebuild === true);
                    sync();
                };
                input = num;
            }
            row.append(name, input);
            panel.appendChild(row);
        }
    };

    render();
    host.appendChild(panel);
    return () => panel.remove();
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
 * per-pair `instanceCount` for one view slot, decoded from a {@link Mirror} of a kitchen cull
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

// Wire `window.__benchmark` — the single contract harness/core drives. `run` measures through
// the profiler (timing source of truth), then asserts; the page treats the returned Verdict as
// opaque and the launcher interprets it. `ready` waits on both the built scene and the
// profiler's first resolved frame.
export function installHarness(scenario: Scenario, state: State, built: () => boolean): void {
    const target: HarnessTarget = {
        get ready() {
            return built() && window.__benchmark?.ready === true;
        },
        async run(opts: RunOpts): Promise<Verdict> {
            const benchmark = window.__benchmark;
            if (!benchmark) {
                throw new Error("ProfilePlugin missing — window.__benchmark not installed");
            }
            const metrics: BenchmarkMeasurement = await benchmark.measure(opts.warmup, opts.frames);
            const checks = await scenario.assert?.(state);
            return checks ? { metrics, checks } : { metrics };
        },
    };
    window.__harness = target;
}
