// ocean-fold — I2r-b's GPU composed-field fold arm: the CPU-only fold-anchor oracle
// (`packages/shallot-ocean/tests/fold-anchor.oracle.ts`) solves λ against the whitecap anchor
// entirely on the CPU reference pipeline. This scenario is the one place that quantity is read off
// the device the render loop actually runs on — `measureFoldFraction` (`@dylanebert/shallot-ocean`)
// runs the ACTUAL production compute pipeline (`createCascadeState` + `encodeCascadePasses`, the
// same functions the per-frame render loop uses) at an arbitrary one-off config and time, off the
// persistent `cascades[]` array, and reads back the per-texel `ProbeData.negDetCount`/`totalCount`
// `postKernel` already writes. COMPOSED here means the same pooled (population-weighted, both
// displacement cascades summed) reading `fold-anchor.oracle.ts` derives λ against — see that file's
// header for why "composed" means cross-cascade pooling rather than per-cascade.
//
// `measureFoldFraction`'s GPU cascade always seeds with `generateH0(config, 0)` (`createCascadeState`'s
// own hardcoded seed, matching the persistent render-loop cascades) — the CPU comparison side below
// uses the identical seed so both sides read the same realization, differing only in which pipeline
// (WGSL f32 compute vs `cpu-reference.ts`'s transcribed CPU arithmetic) computed it.
//
// `bun bench` itself already skips loudly (`scripts/bench.ts`'s `skipReason()`) when the seat has no
// real GPU adapter, before any scenario boots — this file adds no second adapter check.
import { RenderPlugin, run, SlabPlugin, type State, TransformsPlugin } from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    CASCADE_CONFIGS,
    generateH0,
    measureFoldFraction,
    OceanPlugin,
    runCpuPipeline,
} from "@dylanebert/shallot-ocean";
import { type Check, register, type Scenario } from "../gym";

const [CFG0, CFG1] = CASCADE_CONFIGS;
const TOTAL_TEXELS = CFG0.N * CFG0.N + CFG1.N * CFG1.N;

// >= 8 phases, declared before the reading (spec Validation: "reads at least 8 phases"). Fold
// fraction is time-invariant for this stationary field (`mesh-inversion-sweep.oracle.ts`'s own
// header), so any 8 distinct times exercise the same claim; these are plain integer seconds for
// readability, never chosen to land on anything special.
const PHASES = [0, 1, 2, 3, 4, 5, 6, 7];

interface PhaseReading {
    time: number;
    cpuPooled: number;
    gpuPooled: number;
    cpuCount0: number;
    cpuCount1: number;
    gpuCount0: number;
    gpuCount1: number;
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
        readings.push({
            time,
            cpuPooled: cpu.pooled,
            gpuPooled: gpu.pooled,
            cpuCount0: cpu.count0,
            cpuCount1: cpu.count1,
            gpuCount0: gpu.count0,
            gpuCount1: gpu.count1,
        });
    }

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
            name: `t=${r.time}s composed fold: CPU vs GPU`,
            pass: true, // printed reading; the aggregate check below is what gates
            detail:
                `CPU pooled=${(r.cpuPooled * 100).toFixed(4)}% (counts ${r.cpuCount0}+${r.cpuCount1}) ` +
                `GPU pooled=${(r.gpuPooled * 100).toFixed(4)}% (counts ${r.gpuCount0}+${r.gpuCount1})`,
        });
    }

    checks.push({
        name: "CPU and GPU composed fold-texel counts agree across every declared phase",
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
