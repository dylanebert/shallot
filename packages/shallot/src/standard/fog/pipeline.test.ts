import { describe, expect, test } from "bun:test";
import { flat, noIntegerDivision } from "../../../tests/wgsl";
import { fogLayout0, fogLayout1, fogWgsl } from "./pipeline";

// The fog march's typed layouts + kernel, resolved device-free — the `outlineWgsl`/`glazeWgsl` shape. These
// pin the structure the pipeline gates can't cheaply reach: the group-0/group-1 binding shapes, the six
// shadow-only bindings forced into scope despite no direct `layout.$.x` access in the kernel body, and the
// idiv audit (0a's probe found the fog kernel needs none). Numerics are the bench's job
// (`bun bench --scenario render --param mode=fog`); this is what the GPU can't tell us about cheaply.
describe("fog bind-group layouts", () => {
    test("group 0 has the five per-camera entries at distinct indices", () => {
        expect(fogLayout0.index === undefined || typeof fogLayout0.index === "number").toBe(true);
        expect(Object.keys(fogLayout0.entries)).toEqual([
            "sceneTex",
            "depthTex",
            "output",
            "view",
            "fog",
        ]);
    });

    test("group 1 has the ten camera-independent light + shadow entries", () => {
        expect(Object.keys(fogLayout1.entries)).toEqual([
            "pointLights",
            "lightGrid",
            "lightIndices",
            "pointAtlas",
            "pointShadows",
            "shadowSamp",
            "shadowMap",
            "sunShadow",
            "lighting",
            "tileRects",
        ]);
    });
});

describe("fog march kernel", () => {
    const wgsl = fogWgsl();
    const flatWgsl = flat(wgsl);

    test("binds the five group-0 resources by their typed-layout names", () => {
        expect(flatWgsl).toMatch(/var sceneTex: texture_2d<f32>;/);
        expect(flatWgsl).toMatch(/var depthTex: texture_depth_2d;/);
        expect(flatWgsl).toMatch(/var output: texture_storage_2d<rgba16float, write>;/);
        expect(flatWgsl).toMatch(/var<uniform> view: View;/);
        expect(flatWgsl).toMatch(/var<uniform> fog: Fog;/);
    });

    test("binds the ten group-1 resources by their typed-layout names", () => {
        expect(flatWgsl).toMatch(/var<storage, read> pointLights: PointLights;/);
        expect(flatWgsl).toMatch(/var<storage, read> lightGrid: array<vec2u>;/);
        expect(flatWgsl).toMatch(/var<storage, read> lightIndices: array<u32>;/);
        expect(flatWgsl).toMatch(/var pointAtlas: texture_depth_2d;/);
        expect(flatWgsl).toMatch(/var<uniform> pointShadows: PointCasters;/);
        expect(flatWgsl).toMatch(/var shadowSamp: sampler_comparison;/);
        expect(flatWgsl).toMatch(/var shadowMap: texture_depth_2d;/);
        expect(flatWgsl).toMatch(/var<uniform> sunShadow: SunShadow;/);
        expect(flatWgsl).toMatch(/var<uniform> lighting: Lighting;/);
        expect(flatWgsl).toMatch(/var<uniform> tileRects: TileRects;/);
    });

    test("calls the real pointShadowOf / sampleSunShadow references, not a spliced chunk", () => {
        expect(wgsl).toContain(
            "fn pointShadowOf(light: PointLightGpu, normal: vec3f, fragWorld: vec3f)",
        );
        expect(wgsl).toContain("fn sampleSunShadow(worldPos: vec3f, normal: vec3f)");
        expect(flatWgsl).toContain("let shadow = pointShadowOf(light, vec3f(), p);");
        expect(flatWgsl).toContain("sampleSunShadow(p, vec3f())");
    });

    // pointShadowOf/sampleSunShadow read pointAtlas/shadowSamp/pointShadows/tileRects/shadowMap/sunShadow as
    // free names — invisible to tgpu.resolve's call-graph walk from the kernel's own body (the `hullData`
    // forcing-touch precedent), so `forcedZero` is what pulls their bindings into scope. Mutating it away
    // (commenting the fold) reproduces the "did not compile" failure the real pipeline hit before this
    // fold existed — proving the test is load-bearing, not just descriptive.
    test("forces the six shadow-only bindings into scope via the live forcedZero fold", () => {
        expect(flatWgsl).toContain("pointShadows.casters[0i].pos.x");
        expect(flatWgsl).toContain("tileRects.rects[0i].x");
        expect(flatWgsl).toContain("sunShadow.enabled");
        expect(flatWgsl).toContain("textureSampleCompareLevel(pointAtlas, shadowSamp");
        expect(flatWgsl).toContain("textureSampleCompareLevel(shadowMap, shadowSamp");
        // folded through `t`, which is what `fogComposite` actually reads — never a discarded local
        expect(flatWgsl).toContain("let t = (trans + forcedZero);");
        expect(flatWgsl).toContain("fogComposite(scn, (*cfg).color.xyz, t)");
    });

    test("the march loop keeps its constant upper bound with a dynamic break (DXC discipline)", () => {
        expect(flatWgsl).toContain("while ((i < 256u)) {");
        expect(flatWgsl).toContain("if ((i >= steps)) { break; }");
    });

    test("the accumulator terms keep their operand order — no reassociation", () => {
        // (trans·albedo·gain) as one scalar product scaling lstep, THEN scaled again by (1−sampleTrans) —
        // the shipped shader's left-to-right grouping; folding the last factor into the leading scalar
        // chain first changes the f32 rounding
        expect(flatWgsl).toContain(
            "inScatter = (inScatter + ((lstep * ((trans * albedo) * gain)) * (1f - sampleTrans)));",
        );
    });

    test("no integer division — the fog kernel's every division is real f32 (dist/steps, seg/max(dist,ε))", () => {
        noIntegerDivision(wgsl);
    });
});
