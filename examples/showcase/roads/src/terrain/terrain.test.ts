import { describe, expect, test } from "bun:test";
import { body, flat } from "../../../../../packages/shallot/tests/wgsl";
import { DIST_RANGE, TILE_SIZE, TILES_PER_SIDE } from "../overlay/tiles";
import { terrainFsWgsl } from "./terrain";

// The overlay composite's structural gate — the device-free seam this stage's `bun test` relies on for the
// fs's atlas-sampling half (`testing.md`'s ladder: CPU-callable/resolved-WGSL first, never a bound
// device). The composite's actual pixel output is the capture gate's own arm (`gate.ts`/`test/roads.spec.ts`)
// — checks.md's "layers are a granularity": the compute-write half is proven by the seeded-tile readback
// oracle (`overlay/stroke.test.ts`), this half by resolution + the capture, neither alone.

describe("terrain fs — overlay composite", () => {
    const wgsl = terrainFsWgsl();

    test("declares the four overlay bindings the surface layout adds", () => {
        expect(flat(wgsl)).toContain(
            "@group(2) @binding(3) var<storage, read> indirection: array<i32>;",
        );
        expect(wgsl).toContain("var albedo: texture_2d_array<f32>;");
        expect(wgsl).toContain("var dist: texture_2d_array<f32>;");
        expect(wgsl).toContain("var overlaySamp: sampler;");
    });

    test("addresses the tile grid from world (x, z) using the documented constants", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain(
            `(ctx.world.x + ${TILE_SIZE * (TILES_PER_SIDE / 2)}f) / ${TILE_SIZE}f`,
        );
        expect(fs).toContain(
            `(ctx.world.z + ${TILE_SIZE * (TILES_PER_SIDE / 2)}f) / ${TILE_SIZE}f`,
        );
        expect(fs).toMatch(/clamp\(i32\(floor\(.*\)\), 0i, \d+i\)/);
        expect(fs).toContain(`(tz * ${TILES_PER_SIDE}i) + tx`);
    });

    test("looks the tile up in the indirection buffer, never inside a per-fragment branch around the sample", () => {
        // textureSample computes screen-space derivatives, so WGSL requires uniform control flow — a
        // per-fragment `if (layer >= 0)` guard around it is a real-GPU compile rejection ("must only be
        // called from uniform control flow"), caught red-handed by the device gate. The fix is unconditional
        // sampling + a `select`-masked contribution instead of a branch; this test pins that shape stays.
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("let layer = indirection[u32(tileIdx)];");
        expect(fs).not.toMatch(/if\s*\(\(layer/);
        expect(fs).toContain("let layerU = u32(max(layer, 0i));");
        expect(fs).toMatch(/select\(0f, 1f, \(layer >= 0i\)\)/);
    });

    test("samples both atlas arrays unconditionally, by the same tile-local uv and clamped layer", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("textureSample(dist, overlaySamp, uv, layerU)");
        expect(fs).toContain("textureSample(albedo, overlaySamp, uv, layerU)");
    });

    test("decodes the distance channel with the documented DIST_RANGE, thresholds coverage with fwidth, and masks it by residency", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain(`* ${DIST_RANGE}f`);
        expect(fs).toContain("fwidth(");
        // Green 2007's alpha-tested-magnification form, masked by the residency select above (an
        // unallocated tile's coverage is forced to 0 regardless of what garbage the clamped-layer sample
        // returns) — never a branch, per the uniform-control-flow test above.
        expect(fs).toMatch(/clamp\(\(0\.5f - \(dist_\d* \/ fw\)\), 0f, 1f\) \* resident/);
    });

    test("composites the overlay over the base color by coverage — never a second draw", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("color = mix(color, overlay, coverage);");
        expect(wgsl.match(/fn terrainFs\(/g)?.length).toBe(1); // exactly one fs entry — no second pass
    });
});
