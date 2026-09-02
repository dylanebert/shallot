// ocean-fold — I2r-b's GPU composed-field fold arm: the CPU-only fold-anchor oracle
// (`packages/shallot-ocean/tests/fold-anchor.oracle.ts`) solves λ against the whitecap anchor
// entirely on the CPU reference pipeline. This scenario is the one place that quantity is read off
// the device the render loop actually runs on — `measureFoldFraction` (`@dylanebert/shallot-ocean`)
// runs the ACTUAL production compute pipeline (`createCascadeState` + `encodeCascadePasses`, the
// same functions the per-frame render loop uses) at an arbitrary one-off config and time, off the
// persistent `cascades[]` array, and reads back the per-texel `ProbeData.negDetCount`/`totalCount`
// `postKernel` already writes.
//
// TWO DIFFERENT "COMPOSED" READINGS, on purpose, per the spec's own escape hatch (`fold-anchor.
// oracle.ts`'s composed-world-grid superposition has no GPU-side counterpart — `measureFoldFraction`
// reads each cascade's OWN native texel grid, no cross-cascade world-point interpolation or
// composition on the GPU):
//   (1) SUBSTRATE agreement (per-cascade Jacobian, CPU vs GPU) — `cpuPooledFold`/`gpuPooledFold`
//       below, comparing each cascade's own closed-form det-J texel count between the two pipelines.
//       This is the arm that actually gates CPU/GPU agreement in this file.
//   (2) COMPOSED statistic (world-grid-superposed, CPU-only) — `composeWorldGrid`/`foldFractionAt`
//       (`@dylanebert/shallot-ocean`, the exact functions `fold-anchor.oracle.ts` solves λ with),
//       read and printed at the same >= 8 phases, never compared to a GPU reading since none exists
//       to compare it to. This is the statistic `fold-anchor.oracle.ts` actually gates λ against;
//       this scenario reads it here only to confirm it stays reachable off the device-agnostic CPU
//       reference at these phases, not to re-derive λ.
//
// `measureFoldFraction`'s GPU cascade always seeds with `generateH0(config, 0)` (`createCascadeState`'s
// own hardcoded seed, matching the persistent render-loop cascades) — every CPU reading below (both
// the substrate and the composed one) uses the identical seed so every side reads the same
// realization, differing only in which pipeline (WGSL f32 compute vs `cpu-reference.ts`'s
// transcribed CPU arithmetic) computed it, or whether cascades are pooled per-texel or composed on
// one world grid.
//
// `bun bench` itself already skips loudly (`scripts/bench.ts`'s `skipReason()`) when the seat has no
// real GPU adapter, before any scenario boots — this file adds no second adapter check.
import { RenderPlugin, run, SlabPlugin, type State, TransformsPlugin } from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    CASCADE_CONFIGS,
    type CascadeGradientField,
    composeWorldGrid,
    foldFractionAt,
    generateH0,
    measureFoldFraction,
    OceanPlugin,
    realPart,
    runCpuPipeline,
    worldGridSpec,
} from "@dylanebert/shallot-ocean";
import { type Check, register, type Scenario } from "../gym";

const [CFG0, CFG1] = CASCADE_CONFIGS;
const TOTAL_TEXELS = CFG0.N * CFG0.N + CFG1.N * CFG1.N;
const WORLD_GRID = worldGridSpec(CASCADE_CONFIGS);

// >= 8 phases, declared before the reading (spec Validation: "reads at least 8 phases"). Fold
// fraction is time-invariant for this stationary field (`mesh-inversion-sweep.oracle.ts`'s own
// header), so any 8 distinct times exercise the same claim; these are plain integer seconds for
// readability, never chosen to land on anything special.
const PHASES = [0, 1, 2, 3, 4, 5, 6, 7];

/** The composed-world-grid reading at one phase, seed 0 (matching `measureFoldFraction`'s own
 *  hardcoded GPU seed) — CPU-only, no GPU comparison (this file's own header). */
function cpuComposedFold(time: number): number {
    const fieldFor = (cfg: (typeof CASCADE_CONFIGS)[number]): CascadeGradientField => {
        const h0 = generateH0(cfg, 0);
        const cpu = runCpuPipeline(h0, cfg, time);
        return {
            N: cfg.N,
            L: cfg.L,
            gxx: realPart(cpu.gxxHeight, cfg.N),
            gxz: realPart(cpu.gxzHeight, cfg.N),
            gzz: realPart(cpu.gzzHeight, cfg.N),
        };
    };
    const composed = composeWorldGrid([fieldFor(CFG0), fieldFor(CFG1)], WORLD_GRID);
    return foldFractionAt(composed, CFG0.lambda);
}

interface PhaseReading {
    time: number;
    cpuPooled: number;
    gpuPooled: number;
    cpuCount0: number;
    cpuCount1: number;
    gpuCount0: number;
    gpuCount1: number;
    /** the composed-world-grid reading (CPU-only — this file's own header). */
    cpuComposed: number;
}

function cpuPooledFold(time: number): { pooled: number; count0: number; count1: number } {
    const h0Cfg0 = generateH0(CFG0, 0);
    const h0Cfg1 = generateH0(CFG1, 0);
    const cpu0 = runCpuPipeline(h0Cfg0, CFG0, time);
    const cpu1 = runCpuPipeline(h0Cfg1, CFG1, time);
    return {
        pooled: (cpu0.jacobian.foldCount + cpu1.jacobian.foldCount) / TOTAL_TEXELS,
        count0: cpu0.jacobian.foldCount,
        count1: cpu1.jacobian.foldCount,
    };
}

async function gpuPooledFold(
    time: number,
): Promise<{ pooled: number; count0: number; count1: number }> {
    const [frac0, frac1] = await Promise.all([
        measureFoldFraction(CFG0, time),
        measureFoldFraction(CFG1, time),
    ]);
    // measureFoldFraction returns negDetCount/totalCount; totalCount === N*N exactly (postKernel
    // writes it unconditionally, one per texel), so recovering the count is exact, not a rounding.
    const count0 = Math.round(frac0 * CFG0.N * CFG0.N);
    const count1 = Math.round(frac1 * CFG1.N * CFG1.N);
    return { pooled: (count0 + count1) / TOTAL_TEXELS, count0, count1 };
}

async function runChecks(_state: State): Promise<Check[]> {
    const checks: Check[] = [];

    checks.push({
        name: `${PHASES.length} declared phases (>= 8 required)`,
        pass: PHASES.length >= 8,
        detail: `phases=[${PHASES.join(", ")}]`,
    });

    const readings: PhaseReading[] = [];
    for (const time of PHASES) {
        const cpu = cpuPooledFold(time);
        const gpu = await gpuPooledFold(time);
        const cpuComposed = cpuComposedFold(time);
        readings.push({
            time,
            cpuPooled: cpu.pooled,
            gpuPooled: gpu.pooled,
            cpuCount0: cpu.count0,
            cpuCount1: cpu.count1,
            gpuCount0: gpu.count0,
            gpuCount1: gpu.count1,
            cpuComposed,
        });
    }

    checks.push({
        name: `composed-world-grid reading is reachable off the CPU reference at every declared phase (${WORLD_GRID.gridN}x${WORLD_GRID.gridN} points, spacing=${WORLD_GRID.spacing.toFixed(4)}m, extent=${WORLD_GRID.extent.toFixed(2)}m)`,
        pass: readings.every((r) => Number.isFinite(r.cpuComposed) && r.cpuComposed >= 0),
        detail: readings
            .map((r) => `t=${r.time}s composed=${(r.cpuComposed * 100).toFixed(4)}%`)
            .join(", "),
    });

    // Per-phase, per-cascade texel-count agreement: CPU and GPU compute the SAME closed-form
    // Jacobian arithmetic (no reconstruction kernel, no discretization choice — see this stage's
    // Locked-decision boundary) from the same seed, so the only source of disagreement is
    // floating-point evaluation order/precision near det J == 0. Zero disagreement is the norm at
    // this SNR (fold fraction is a low-single-digit percentage of 20480 texels); a nonzero count
    // is printed exactly, never masked by an authored percentage tolerance.
    let totalCpuCount = 0;
    let totalGpuCount = 0;
    let maxAbsCascadeDelta = 0;
    for (const r of readings) {
        totalCpuCount += r.cpuCount0 + r.cpuCount1;
        totalGpuCount += r.gpuCount0 + r.gpuCount1;
        maxAbsCascadeDelta = Math.max(
            maxAbsCascadeDelta,
            Math.abs(r.cpuCount0 - r.gpuCount0),
            Math.abs(r.cpuCount1 - r.gpuCount1),
        );
        checks.push({
            name: `t=${r.time}s substrate (per-cascade pooled) fold: CPU vs GPU`,
            pass: true, // printed reading; the aggregate check below is what gates
            detail:
                `CPU pooled=${(r.cpuPooled * 100).toFixed(4)}% (counts ${r.cpuCount0}+${r.cpuCount1}) ` +
                `GPU pooled=${(r.gpuPooled * 100).toFixed(4)}% (counts ${r.gpuCount0}+${r.gpuCount1}) ` +
                `— CPU composed (world-grid, no GPU counterpart)=${(r.cpuComposed * 100).toFixed(4)}%`,
        });
    }

    checks.push({
        name: "CPU and GPU substrate (per-cascade Jacobian) fold-texel counts agree across every declared phase",
        pass: maxAbsCascadeDelta === 0,
        detail:
            `totalCpuCount=${totalCpuCount} totalGpuCount=${totalGpuCount} ` +
            `maxAbsPerCascadeDelta=${maxAbsCascadeDelta} over ${PHASES.length} phases x 2 cascades`,
    });

    return checks;
}

let checks: Check[] = [];

const scenario: Scenario = {
    name: "ocean-fold",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, SlabPlugin, TransformsPlugin, RenderPlugin, OceanPlugin],
        });
        state.pause();
        try {
            checks = await runChecks(state);
        } catch (err) {
            checks = [{ name: "ocean-fold runChecks threw", pass: false, detail: String(err) }];
        }
        return { state, dispose };
    },
    async assert() {
        return checks;
    },
};

register(scenario);
