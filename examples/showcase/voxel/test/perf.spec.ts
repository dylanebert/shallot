import { expect, test } from "@playwright/test";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// The structural perf gate: a wall-clock frame-time gate is refuted for this mechanism (
// the display clock dominates both the idle-orbit and carve-drag windows on this seat's 240 Hz monitor, so
// any ratio over frame time or `fenceMs` gates the monitor, not the code). The admissible form is a
// structural gate on the named mechanism — emit dispatch scope (workgroup count per emit fire) as a
// function of touched-chunk volume, derived below rather than fitted.
//
// Derivation of the bound: the emit kernel (`mesher.ts`'s `emitKernel`) decides each of a cell's six faces
// by reading exactly one neighbour cell, one axis at a time (`ix±1`, `iy±1`, `iz±1`). A cell sitting on a
// chunk's boundary layer can therefore only pull a face-culling read across ONE face-adjacent chunk per
// axis — never a diagonal (edge/corner) chunk, since the three axis offsets are independent and never
// combine. Correctly re-meshing a touched chunk needs at most that chunk plus its 6 face-adjacent
// neighbours dispatched — never the other 20 chunks of a full 3×3×3 halo, and never the grid beyond it.
// `HALO_CHUNKS` below names that count; `CHUNK_WORKGROUPS` is one chunk's own dispatch volume in the
// kernel's real `WG`-sized workgroups (`(CHUNK / WG) ** 3`, both read from `src/voxel/*` — not restated).
//
// Per-fire wall-clock occupancy is printed for context, never gated: the same reasoning that refutes
// a frame-time gate refutes a GPU-occupancy-in-ms gate too (no arbitrary numerical band). Forced over
// FORCE_FIRES real fires — a small handful of samples gates nothing, so a quoted percentile here needs
// enough of them to mean something.
//
// Witnessed red at the pre-fix ref (2026-08-24, RTX 4090 via the host bridge): exit 1 — "dispatched 262144
// workgroups for a 1-chunk edit — expected <= 3584 (touched-chunk-scoped dispatch); the pre-fix full-grid
// dispatch is always 262144", exactly the full-grid dispatch this bound replaces. The correctness gate
// (`test/voxel.spec.ts`) and the project's 51 unit tests stay green against the same tree.
//
// A naive fix that narrows the dispatch alone is refuted: the shared atomic-append buffers only stay
// complete if every chunk in a dispatch re-emits together, so a scoped dispatch needs a per-chunk
// buffer-region/compaction rewrite underneath it — chunk table, per-chunk cursors, a CPU-exact allocator
// (`mesher.ts`, `chunkpool.ts`). The bound derivation and the witnessed-red record above are the fixture
// that mechanism satisfies.

import { CHUNK } from "../src/voxel/grid";
import { WG } from "../src/voxel/mesher";

const CHUNK_WORKGROUPS = (CHUNK / WG) ** 3; // one chunk's own emit-kernel dispatch volume (512)
const HALO_CHUNKS = 7; // the touched chunk + its 6 face-adjacent neighbours (derivation above)
const FORCE_FIRES = 20; // enough repeated fires for a stable percentile reading, cheap at one chunk per fire

test("structural — emit dispatch scope tracks touched-chunk volume, not the full grid", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`voxel perf gate adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    await page.waitForFunction(
        () => typeof window.__voxelPerf === "function" && window.__benchmark?.ready === true,
        null,
        { timeout: 120_000 },
    );

    // the structural sample: one deterministic single-chunk edit, dispatch scope read back.
    const sample = await page.evaluate(() => window.__voxelPerf!());
    expect(sample.touchedChunks, "fixture drift — the probe must flip exactly one chunk").toBe(1);

    // ungated: per-fire GPU occupancy over FORCE_FIRES real fires, printed for context only — never a
    // gated quantity (no arbitrary wall-clock band). Read and printed *before* the
    // structural assertion below so a red gate still leaves the diagnostic on the record.
    const occ = await page.evaluate(async (fires: number) => {
        const bench = window.__benchmark!;
        const measured = bench.measure(0, fires * 6 + 30);
        for (let i = 0; i < fires; i++) await window.__voxelPerf!();
        const stats = await measured;
        return stats.gpu?.passes["voxel:emit"] ?? null;
    }, FORCE_FIRES);
    console.log(
        occ
            ? `voxel:emit occupancy over ${FORCE_FIRES} fires (printed only, never gated): ` +
                  `occMs=${occ.occMs} occP95=${occ.occP95} occP99=${occ.occP99}`
            : "voxel:emit occupancy: no samples drained in the measured window",
    );

    const bound = sample.touchedChunks * HALO_CHUNKS * CHUNK_WORKGROUPS;
    console.log(
        `voxel emit dispatch: ${sample.workgroups} workgroups for ${sample.touchedChunks} touched ` +
            `chunk(s) (structural bound ${bound} = touchedChunks × ${HALO_CHUNKS} halo chunks × ` +
            `${CHUNK_WORKGROUPS} workgroups/chunk)`,
    );
    expect(errors, errors.join("\n")).toEqual([]);
    expect(
        sample.workgroups,
        `dispatched ${sample.workgroups} workgroups for a ${sample.touchedChunks}-chunk edit — ` +
            `expected <= ${bound} (touched-chunk-scoped dispatch); the pre-fix full-grid dispatch is ` +
            `always 262144`,
    ).toBeLessThanOrEqual(bound);
});
