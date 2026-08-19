import { decodePos } from "@dylanebert/shallot/utils/core";
import { meshHeightAt } from "./capture";
import {
    buildDeviceFreeVertices,
    checkSurfaceFlatness,
    RECONSTRUCTION_AGREEMENT_TOL,
    reconstructionAgreement,
} from "./flatness";
import type { Check } from "./harness";
import { generateNetwork } from "./overlay/network";
import { TERRAIN_QUANT } from "./terrain/generate";
import { VERTEX_COUNT } from "./terrain/grid";
import { RELIEF } from "./terrain/noise";
import {
    generate,
    getSmoothRadius,
    readVertices,
    SEED,
    syncNetworkForSeed,
} from "./terrain/terrain";

// The terrain generator's correctness gate — the seed-determinism readback the spec's Validation names
// ("Overlay correctness"/"Rasterizer fidelity" are later stages; this stage's own arm is the height
// kernel's seed determinism, run on the real device). It's the showcase dogfooding its own testing:
// published-`@dylanebert/shallot` surface + this project's own lib + driver, no reach into any repo
// harness. `boot.ts` exposes it on `window.__roadsGate`; the project's own Playwright
// (`test/roads.spec.ts`) drives it on a GPU.
//
// `bun test ./src` (noise.test.ts, grid.test.ts, generate.test.ts) covers the device-free half: the
// permutation table's seed determinism, the WGSL structural resolve, and the grid-topology oracle —
// `bun test` never binds a device. This gate covers the one thing
// only a real device can show: that two GPU dispatches at the same seed actually produce byte-identical
// vertex content, and two different seeds don't.

const GATE_SEED_A = 0x0072_6f61; // "roa" — a fixed seed independent of the live terrain's SEED (boot.ts)
const GATE_SEED_B = 0x0064_7332; // "ds2"

/** the per-column surface-height standard deviation, decoded from the raw vertex stream via the
 *  published {@link decodePos} — the same decode sear's vertex pull uses, so this reads exactly what the
 *  renderer will read. Samples every 8th vertex (a stride, not the full 66k) since the stat only needs
 *  enough points to detect "flat vs rolling", not a full census. */
function heightStd(raw: Uint32Array): number {
    const stride = 8;
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    for (let i = 0; i < VERTEX_COUNT; i += stride) {
        const w0 = raw[i * 4];
        const w1 = raw[i * 4 + 1];
        const y = decodePos(w0, w1, TERRAIN_QUANT).y;
        sum += y;
        sumSq += y * y;
        n++;
    }
    const mean = sum / n;
    return Math.sqrt(Math.max(0, sumSq / n - mean * mean));
}

function equalStream(a: Uint32Array, b: Uint32Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

/** run the terrain generator's gate against the live device, restoring the boot seed's terrain when done. */
export async function gate(): Promise<Check[]> {
    const checks: Check[] = [];

    // the flatten targets are baked CPU-side per seed (`terrain.ts`'s `syncNetworkForSeed`) — keep them in
    // step with each probe seed's own permutation before dispatching, or the road's plateau would blend
    // toward a stale seed's terrain under this seed's natural surface.
    syncNetworkForSeed(GATE_SEED_A);
    await generate(GATE_SEED_A);
    const first = await readVertices();
    checks.push({
        name: "vertex-stream-size",
        pass: first.length === VERTEX_COUNT * 4,
        detail: `${first.length} u32 words (expected ${VERTEX_COUNT * 4})`,
    });

    await generate(GATE_SEED_A);
    const repeat = await readVertices();
    checks.push({
        name: "deterministic",
        pass: equalStream(first, repeat),
        detail: "seed → bit-identical vertex stream across two runs",
    });

    syncNetworkForSeed(GATE_SEED_B);
    await generate(GATE_SEED_B);
    const reseeded = await readVertices();
    checks.push({
        name: "reseed-changes",
        pass: !equalStream(first, reseeded),
        detail: "a different seed produces a different vertex stream",
    });

    // the relief floor: a 5-octave zero-mean fbm's surface std is ≳ 0.1·RELIEF (voxel's `noise.ts`
    // derives the same bound for its heightmap — the octave/persistence pair is identical here), so
    // halving it for seed-to-seed variation gives a not-flat floor with ~2× margin, not a tuned threshold.
    const std = heightStd(reseeded);
    const reliefFloor = RELIEF * 0.05;
    checks.push({
        name: "relief",
        pass: std > reliefFloor,
        detail: `surface-height std ${std.toFixed(2)} > ${reliefFloor.toFixed(2)} (rolling, not flat)`,
    });

    // restore the boot seed's terrain for the live view.
    syncNetworkForSeed(SEED);
    await generate(SEED);

    // Stage 15c's device arm — re-pinned on the property (the surface-flatness readings on the real
    // device's mesh), not just the fidelity of the CPU lattice against the GPU. The fidelity check
    // (`reconstructionAgreement`) stays — it proves `bun test`'s device-free arm reads the same mesh the
    // real device renders, without which a CPU/GPU divergence could make the device-free suite pass or
    // fail for the wrong reason. But 15c also runs `checkSurfaceFlatness` on the real `readVertices()`
    // and asserts the out-of-zone violation counts match the device-free lattice's counts — the property
    // itself reads the same on both sides, not just the heights. No absolute amplitude bound is asserted
    // (Blocker 3: any honest bound is ≥ MAX_GRADE·SPACING and reads stage 12's founding defect green);
    // the convergence ratio (Leg B) needs two resolutions and lives in the device-free suite.
    const liveDoc = generateNetwork(SEED);
    const deviceRaw = await readVertices();
    const agreement = reconstructionAgreement(deviceRaw, liveDoc, SEED, getSmoothRadius());
    checks.push({
        name: "surface-flatness-reconstruction-agreement",
        pass: agreement.maxDiffM <= RECONSTRUCTION_AGREEMENT_TOL,
        detail: `CPU device-free lattice vs real readVertices(): max |Δh| ${agreement.maxDiffM.toFixed(6)} m over ${agreement.sampleCount} footprint samples (tol ${RECONSTRUCTION_AGREEMENT_TOL.toFixed(6)} m)`,
    });

    // the property pin: the real device's mesh reads the same out-of-zone violation counts as the
    // device-free lattice (the flatness property, not just point-by-point height fidelity).
    const cpuRaw = buildDeviceFreeVertices(liveDoc, SEED, getSmoothRadius());
    const deviceResult = checkSurfaceFlatness((x, z) => meshHeightAt(deviceRaw, x, z), liveDoc);
    const cpuResult = checkSurfaceFlatness((x, z) => meshHeightAt(cpuRaw, x, z), liveDoc);
    checks.push({
        name: "surface-flatness-property-on-device",
        pass:
            deviceResult.crossSection.length === cpuResult.crossSection.length &&
            deviceResult.longitudinal.length === cpuResult.longitudinal.length &&
            Math.abs(deviceResult.maxCrossSectionExcess - cpuResult.maxCrossSectionExcess) <
                RECONSTRUCTION_AGREEMENT_TOL,
        detail: `device crossSection=${deviceResult.crossSection.length} longitudinal=${deviceResult.longitudinal.length} maxExcess=${deviceResult.maxCrossSectionExcess.toFixed(4)} vs CPU crossSection=${cpuResult.crossSection.length} longitudinal=${cpuResult.longitudinal.length} maxExcess=${cpuResult.maxCrossSectionExcess.toFixed(4)}`,
    });

    return checks;
}
