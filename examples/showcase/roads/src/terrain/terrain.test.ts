import { describe, expect, test } from "bun:test";
import { body, flat } from "../../../../../packages/shallot/tests/wgsl";
import { COVERAGE_BAND_PX, DIST_RANGE, TILE_SIZE, TILES_PER_SIDE } from "../overlay/tiles";
import { terrainFsWgsl } from "./terrain";

// The overlay composite's structural gate — the device-free seam this stage's `bun test` relies on for the
// fs's atlas-sampling half (CPU-callable/resolved-WGSL first, never a bound device). The composite's
// actual pixel output is the capture gate's own arm (`gate.ts`/`test/roads.spec.ts`) — layers are a
// granularity: the compute-write half is proven by the seeded-tile readback oracle (`overlay/stroke.test.ts`),
// this half by resolution + the capture, neither alone.

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
        // the coverage band's own coefficient, not just that fwidth appears: capture.ts derives
        // TRANSITION_TOLERANCE_PX as a multiple of COVERAGE_BAND_PX, so a shader-side drift in this
        // factor silently widens the band the capture gate's tolerance was sized against.
        expect(fs).toMatch(new RegExp(`fwidth\\(dist_\\d*\\) \\* ${COVERAGE_BAND_PX}f`));
        // Green 2007's alpha-tested-magnification form, masked by the residency select above (an
        // unallocated tile's coverage is forced to 0 regardless of what garbage the clamped-layer sample
        // returns) — never a branch, per the uniform-control-flow test above.
        expect(fs).toMatch(/clamp\(\(0\.5f - \(dist_\d* \/ fw\)\), 0f, 1f\) \* resident/);
    });

    test("composites the overlay (with markings mixed in) over the base color by coverage — never a second draw", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("color = mix(color, overlayWithMarkings, coverage);");
        expect(wgsl.match(/fn terrainFs\(/g)?.length).toBe(1); // exactly one fs entry — no second pass
    });
});

describe("terrain fs — marking channel (stage 3)", () => {
    const wgsl = terrainFsWgsl();
    const fs = flat(body(wgsl, "fn terrainFs"));

    test("decodes the albedo alpha byte as a signed marking distance using the same DIST_RANGE codec", () => {
        // the alpha byte stores the encoded marking distance (rasterize.ts's encodeDistGpu, the coverage
        // channel's own codec). The fs decodes it back: (sampled.w - 0.5) * 2 * DIST_RANGE — the same
        // decode the coverage channel uses, applied to the albedo's w component, not the dist texture.
        // Anchored on the albedoSample expression (not just `* ${DIST_RANGE}f` alone, which the coverage
        // decode also emits — a bare toContain is blind to its own subject, satisfied by the wrong channel).
        expect(fs).toMatch(new RegExp(`albedoSample.*\\.w.*0\\.5f.*2f.*${DIST_RANGE}f`));
    });

    test("thresholds the marking distance with a second fwidth, exactly as coverage is", () => {
        // a second fwidth call on the marking distance, with the same COVERAGE_BAND_PX coefficient —
        // sub-texel crisp lines at any zoom (the property the Locked decision sells)
        expect(fs).toMatch(new RegExp(`fwidth\\(markingDist_?\\d*\\) \\* ${COVERAGE_BAND_PX}f`));
        // the marking coverage formula mirrors the coverage formula: clamp(0.5 - dist/fw, 0, 1) * resident
        expect(fs).toMatch(
            /clamp\(\(0\.5f - \(markingDist_?\d* \/ markingFw_?\d*\)\), 0f, 1f\) \* resident/,
        );
    });

    test("selects the marking albedo by comparing the decoded marking distance with the edge-line distance", () => {
        // the marking class (edge vs centre) is determined by comparing the decoded marking distance with
        // the edge-line distance computed independently from the coverage distance: if the marking distance
        // is smaller, the nearest marking is the centreline; otherwise the edge line.
        // 0.3 is EDGE_INSET and 0.05 is LINE_HALF_WIDTH — both deliberately literals here, not derived
        // from the exported constants: if the regex were built from EDGE_INSET/LINE_HALF_WIDTH, changing
        // the constant would change both sides and the arm would stay green, asserting only that the
        // shader emitted *some* number consistent with itself, not the *right* number. The literals
        // freeze the derived quantity so the arm reds exactly when the emitted value moves.
        expect(fs).toMatch(/abs\(\(dist_\d* \+ 0\.3\d*f\)\) - 0\.05\d*f/);
        expect(fs).toMatch(/select\(vec3f\(/);
        expect(fs).toMatch(/isCentre/);
    });

    test("mixes the marking albedo over the road albedo before the terrain composite", () => {
        // the order matters: marking over road first, then road+marking over terrain — so a marking
        // never bleeds outside the road's coverage boundary
        expect(fs).toContain("mix(overlay, markingAlbedo, markingCoverage)");
        expect(fs).toContain("color = mix(color, overlayWithMarkings, coverage);");
    });
});
