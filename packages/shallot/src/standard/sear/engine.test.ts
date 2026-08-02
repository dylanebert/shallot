import { describe, expect, test } from "bun:test";
import { flat, noIntegerDivision } from "../../../tests/wgsl";
import { engineLayout, engineScaffoldWgsl } from "./engine";

// The canonical typed engine substrate: the group-0 layout every typed sear pipeline binds and its shared
// `clusterOf` / `pointFactor` / `lightFactor` / `lit` / `litPbr` refs, resolved device-free. These checks
// pin the exact operand order the migrated pipeline uses.

describe("engineLayout", () => {
    test("the seven pass-invariant group-0 entries, in order — no `vertices` (pass-variant, moves to the surface group at 4a-ii-c)", () => {
        expect(Object.keys(engineLayout.entries)).toEqual([
            "frame",
            "view",
            "lighting",
            "pointLights",
            "lightGrid",
            "lightIndices",
            "meshQuant",
        ]);
    });

    test("is pinned to group 0", () => {
        expect(engineLayout.index).toBe(0);
    });
});

describe("engine scaffold", () => {
    const wgsl = engineScaffoldWgsl();
    const flatWgsl = flat(wgsl);

    // `frame` and `meshQuant` are group-0 entries the scaffold never reads (the vertex decode reads
    // meshQuant, not the shading scaffold), so typegpu's
    // resolve — which emits only what's called — declares none of their WGSL: the layout is a superset
    // of what any one shader declares (the 2b precedent).
    test("binds the five group-0 resources it actually reads, by their typed-layout names", () => {
        expect(flatWgsl).toMatch(/var<uniform> view: View;/);
        expect(flatWgsl).toMatch(/var<uniform> lighting: Lighting;/);
        expect(flatWgsl).toMatch(/var<storage, read> pointLights: PointLights;/);
        expect(flatWgsl).toMatch(/var<storage, read> lightGrid: array<vec2u>;/);
        expect(flatWgsl).toMatch(/var<storage, read> lightIndices: array<u32>;/);
        expect(flatWgsl).not.toContain("var<uniform> frame: Frame;");
        expect(flatWgsl).not.toContain("meshQuant");
    });

    test("declares the four var<private> seams the fragment entry fills by name", () => {
        expect(wgsl).toMatch(/var<private> sunVisibility: f32 = 1f;/);
        expect(wgsl).toMatch(/var<private> fragWorld: vec3f = vec3f\(\);/);
        expect(wgsl).toMatch(/var<private> fragCoord: vec4f = vec4f\(\);/);
        expect(wgsl).toMatch(/var<private> pointScale: f32/);
    });

    test("clusterOf keeps the view-depth branch + clusterCell args", () => {
        expect(flatWgsl).toContain("fn clusterOf() -> u32 {");
        expect(flatWgsl).toContain("var viewZ = (1f / fragCoord.w);");
        expect(flatWgsl).toContain(
            "if ((view.cluster.z < 0.5f)) { viewZ = (near + (fragCoord.z * (far - near))); }",
        );
        expect(flatWgsl).toContain(
            "return clusterCell((fragCoord.x / view.resolution.x), (fragCoord.y / view.resolution.y), viewZ, near, far, u32(view.cluster.w));",
        );
    });

    test("pointFactor: zero-scale early-out, the accumulator's left-to-right factor order — no reassociation", () => {
        expect(flatWgsl).toContain("fn pointFactor(normal: vec3f) -> vec3f {");
        expect(flatWgsl).toContain("if ((pointScale == 0f)) { return sum; }");
        // (((distanceAttenuation * diff) * spotFactor) * pointShadowOf), preserving the migrated grouping
        expect(flatWgsl).toContain(
            "sum = (sum + (light.color.rgb * (((distanceAttenuation(distSq, light.posRange.w, radiusSq) * diff) * spotFactor(light, L)) * pointShadowOf(light, normal, fragWorld))));",
        );
    });

    test("lightFactor: (ambient + (sun*sunVisibility)) + pointFactor, left-to-right", () => {
        expect(flatWgsl).toContain(
            "return (((lighting.ambientColor.rgb * lighting.ambientColor.a) + ((lighting.sunColor.rgb * sun) * sunVisibility)) + pointFactor(normal));",
        );
    });

    test("lit: baseColor * lightFactor(normal)", () => {
        expect(flatWgsl).toContain(
            "fn lit(baseColor: vec3f, normal: vec3f) -> vec3f { return (baseColor * lightFactor(normal)); }",
        );
    });

    test("litPbr keeps the ambient/sun/point accumulator terms left-to-right", () => {
        expect(flatWgsl).toContain("fn litPbr(s: Pbr, normal: vec3f, world: vec3f) -> vec3f {");
        // ((ambient.rgb * ambient.a) * albedo) * occlusion
        expect(flatWgsl).toContain(
            "var radiance = (((lighting.ambientColor.rgb * lighting.ambientColor.a) * s.albedo) * s.occlusion);",
        );
        // (sunColor.rgb * sunVisibility) * brdf(...)
        expect(flatWgsl).toContain(
            "radiance = (radiance + ((lighting.sunColor.rgb * sunVisibility) * brdf(s, normal, V, -(lighting.sunDirection.xyz))));",
        );
        // (distanceAttenuation * spotFactor) * pointShadowOf
        expect(flatWgsl).toContain(
            "let f = ((distanceAttenuation(distSq, light.posRange.w, radiusSq) * spotFactor(light, L)) * pointShadowOf(light, normal, fragWorld));",
        );
        // (light.color.rgb * f) * brdfSphere(...)
        expect(flatWgsl).toContain(
            "radiance = (radiance + ((light.color.rgb * f) * brdfSphere(s, normal, V, L, dist, light.params.x)));",
        );
    });

    test("calls the real pointShadowOf reference, not a spliced chunk", () => {
        expect(wgsl).toContain(
            "fn pointShadowOf(light: PointLightGpu, normal: vec3f, fragWorld: vec3f) -> f32",
        );
    });

    test("no integer division", () => {
        noIntegerDivision(wgsl);
        // litPbr's `toLight / dist` is the scaffold's one division — a vec3f/f32, real float, never
        // reachable as an integer pair (dist is `sqrt(max(...))`)
        expect(flatWgsl).toContain("let L = (toLight / dist);");
    });
});
