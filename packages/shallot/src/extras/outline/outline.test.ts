import { describe, expect, test } from "bun:test";
import { groupByMesh, jfaSteps, maskCode, outlineWgsl } from "./passes";

// The JFA step ladder decides how many fullscreen passes the outline runs (the screen × log(width) cost),
// and the mesh grouping decides the scoped instanced draws. Both are pure CPU logic the GPU passes build
// on; the band itself is validated visually in `bun bench --scenario outline`.
describe("outline jfaSteps", () => {
    test("a power-of-two width is the start step, halving to 1", () => {
        expect(jfaSteps(4)).toEqual([4, 2, 1]);
        expect(jfaSteps(64)).toEqual([64, 32, 16, 8, 4, 2, 1]);
    });

    test("a non-power-of-two width rounds the start up to the next power of two", () => {
        // ceil(log2(5)) = 3 → start 8 → 4 passes (a pixel within 5px is within the 8px first jump)
        expect(jfaSteps(5)).toEqual([8, 4, 2, 1]);
        expect(jfaSteps(3)).toEqual([4, 2, 1]);
    });

    test("width 1 is a single step", () => {
        expect(jfaSteps(1)).toEqual([1]);
    });

    test("width clamps to [1, 64] — zero/negative floors at one pass, oversized caps at 64", () => {
        expect(jfaSteps(0)).toEqual([1]);
        expect(jfaSteps(-5)).toEqual([1]);
        expect(jfaSteps(100)[0]).toBe(64); // clamped to MAX_WIDTH before the ladder
    });

    test("every ladder ends at step 1 and strictly halves", () => {
        for (const w of [2, 7, 16, 33, 64]) {
            const steps = jfaSteps(w);
            expect(steps[steps.length - 1]).toBe(1);
            expect(steps[0]).toBeGreaterThanOrEqual(Math.min(64, w));
            for (let i = 1; i < steps.length; i++) expect(steps[i]).toBe(steps[i - 1] / 2);
        }
    });
});

describe("outline groupByMesh", () => {
    test("groups eids by mesh id, preserving insertion order within each group", () => {
        const meshOf = (eid: number) => ({ 10: 1, 11: 2, 12: 1, 13: 2 })[eid] ?? 0;
        const groups = groupByMesh([10, 11, 12, 13], meshOf);
        expect(groups.get(1)).toEqual([10, 12]);
        expect(groups.get(2)).toEqual([11, 13]);
        expect(groups.size).toBe(2);
    });

    test("a single mesh collapses to one group; empty input is an empty map", () => {
        expect(groupByMesh([5, 6, 7], () => 3).get(3)).toEqual([5, 6, 7]);
        expect(groupByMesh([], () => 0).size).toBe(0);
    });
});

describe("outline maskCode", () => {
    // the occlusion gate discards a highlighted fragment that's behind the visible scene. The engine is
    // reverse-Z (near→1/far→0, depthCompare greater), so "behind" = the fragment's depth is LESS than the
    // nearest stored scene depth. A `>` here (forward-Z) never fires, so an occluded outline draws over the
    // wall — the regression this guards (a reverse-Z site silently inverted).
    test("the occlude variant discards with the reverse-Z comparison (clip.z < scene)", () => {
        const occ = maskCode(true);
        expect(occ).toContain("textureLoad(sceneDepth");
        expect(occ).toMatch(/in\.clip\.z\s*<\s*scene/); // reverse-Z: behind = lesser depth
        expect(occ).not.toMatch(/in\.clip\.z\s*>\s*scene/); // forward-Z comparison = the bug
    });

    test("the plain variant has no depth gate (always-on-top)", () => {
        const plain = maskCode(false);
        expect(plain).not.toContain("sceneDepth");
        expect(plain).not.toContain("discard");
    });
});

// The JFA + composite kernels are TGSL, resolved device-free here (the `packWgsl` / `blitWgsl` shape). What
// these pin is the structure the GPU can't tell us about cheaply: the binding types (a seed field read as
// uint, not float — the f16 1024 bug), the fragment output type, the workgroup tile, and the exact
// float-product association of the band term, which a reassociating edit would silently shift.
describe("outline JFA kernel", () => {
    const { jfa } = outlineWgsl();

    test("a fullscreen-triangle vertex entry feeds a fragment entry writing the uint seed target", () => {
        expect(jfa).toContain("@vertex fn fullscreenVs(@builtin(vertex_index) i: u32)");
        // typegpu types a fragment target as a 4-component vector; the rg16uint attachment consumes xy
        expect(jfa).toMatch(
            /@fragment fn jfaFs\(@builtin\(position\) pos: vec4f\) -> @location\(0\) vec4u/,
        );
        expect(jfa).toContain("return vec4u(best.x, best.y, 0u, 0u);");
    });

    test("the seed field is a uint texture and the step a bare f32 uniform", () => {
        // uint, not f16: pixel-center fractions stop being f16-representable at 1024 (see index.ts)
        expect(jfa).toMatch(/var seed: texture_2d<u32>;/);
        expect(jfa).toMatch(/var<uniform> \w+: f32;/);
        expect(jfa).toContain("let s = i32(");
    });

    test("the 3×3 offsets are signed and every out-of-bounds neighbour is skipped", () => {
        // i32 because the offsets go negative; a u32 loop would wrap at -1 and read the far edge
        expect(jfa).toContain("for (var oy = -1i; (oy <= 1i); oy = (oy + 1i))");
        expect(jfa).toContain("for (var ox = -1i; (ox <= 1i); ox = (ox + 1i))");
        // an out-of-bounds textureLoad returns 0, which would read as a seed at the corner
        expect(jfa).toContain(
            "if (((((q.x < 0i) || (q.y < 0i)) || (q.x >= dim.x)) || (q.y >= dim.y))) {",
        );
        expect(jfa).toContain("continue;");
    });

    test("the nearest-seed test keeps the candidate only when strictly closer", () => {
        expect(jfa).toContain("let dist = distance(here, vec2f(cand));");
        expect(jfa).toMatch(/if \(\(dist < bestD\)\) \{\s*bestD = dist;\s*best = cand;\s*\}/);
    });
});

describe("outline composite kernel", () => {
    const { composite } = outlineWgsl();

    test("an 8×8 tile guarded by the output dimensions", () => {
        expect(composite).toContain(
            "@compute @workgroup_size(8, 8) fn compositeKernel(@builtin(global_invocation_id) gid: vec3u)",
        );
        expect(composite).toContain("let dim = textureDimensions(output);");
        expect(composite).toMatch(
            /if \(\(\(gid\.x >= dim\.x\) \|\| \(gid\.y >= dim\.y\)\)\) \{\s*return;/,
        );
    });

    test("binds the scene + seed + attr reads and the rgba16float storage write", () => {
        expect(composite).toMatch(/var scene: texture_2d<f32>;/);
        expect(composite).toMatch(/var seed: texture_2d<u32>;/);
        expect(composite).toMatch(/var attr: texture_2d<f32>;/);
        expect(composite).toMatch(/var output: texture_storage_2d<rgba16float, write>;/);
    });

    test("the band term keeps its float association and its width>0 gate", () => {
        // a * (1 - b), NOT (a - a*b): JS left-to-right reassociation would change the emitted product,
        // and the band is compared against a pre-port framebuffer probe
        expect(composite).toContain(
            "let alpha = (smoothstep(0f, 1f, dist) * (1f - smoothstep((width - 1f), width, dist)));",
        );
        // width 0 (the sentinel seed's out-of-bounds attr read) must yield no band at all
        expect(composite).toContain("let band = select(0f, alpha, (width > 0f));");
        expect(composite).toContain("textureStore(output, p, vec4f(mix(base, a.xyz, band), 1f));");
    });
});
