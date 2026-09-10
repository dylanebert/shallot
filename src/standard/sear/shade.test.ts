import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import {
    brdf,
    brdfSphere,
    CASCADE_FLOATS,
    casterWgsl,
    fresnelSchlick,
    halfLambert,
    Pbr,
    POINT_CASTER_FLOATS,
    pbrWgsl,
    pointShadowWgsl,
    SHADOW_PARAMS_BYTES,
    SUN_PARAMS,
    sunShadowWgsl,
    sunStructWgsl,
    viewDepth,
} from "./shade";
import { MAX_CASCADES } from "./shadows";

// the f32 index the SunShadow globals tail starts at — the cascade array's length, which is what the
// staging writer's absolute tail indices must equal
const TAIL = MAX_CASCADES * CASCADE_FLOATS;

// The shading substrate's TGSL half runs on the CPU, so the invariants its comments claim are testable
// here rather than asserted in prose: the half-Lambert remap, the sphere BRDF reducing to the punctual
// one, and the flat engine default (dielectric 0) meaning literally zero specular. The GPU twin is the
// same function — there is no second implementation to drift.

const flat = (albedo: [number, number, number]) =>
    Pbr({
        albedo: d.vec3f(...albedo),
        metallic: 0,
        roughness: 1,
        occlusion: 1,
        dielectric: 0,
    });

describe("half-Lambert", () => {
    test("the remap wraps the whole sphere and squares", () => {
        expect(halfLambert(1)).toBe(1); // facing the light
        expect(halfLambert(0)).toBe(0.25); // the terminator lifts off zero — the point of the wrap
        expect(halfLambert(-1)).toBe(0); // straight away: dark, so form is kept
        // squared, not the bare remap: the midpoint of the wrapped range sits below its linear value
        expect(halfLambert(-0.5)).toBeLessThan(0.25);
    });
});

describe("metallic-roughness BRDF", () => {
    // the engine `Material` default (metallic 0, roughness 1, dielectric 0) is documented to reduce
    // `litPbr` to the diffuse `lit` exactly — F0 = 0 gives f90 = 0, which kills specular at every angle
    test("the flat default has zero specular, so it is the diffuse lobe alone", () => {
        const s = flat([0.8, 0.4, 0.2]);
        const N = d.vec3f(0, 1, 0);
        const L = d.vec3f(0, 1, 0);
        for (const V of [d.vec3f(0, 1, 0), d.vec3f(0.99, 0.14, 0), d.vec3f(0, 0.1, 0.99)]) {
            const r = brdf(s, N, V, L);
            const want = halfLambert(1); // dot(N, L) = 1
            // the only fp slack is the /PI then *PI round trip inside brdf — a relative f32 ulp
            expect(r.x).toBeCloseTo(0.8 * want, 6);
            expect(r.y).toBeCloseTo(0.4 * want, 6);
            expect(r.z).toBeCloseTo(0.2 * want, 6);
        }
    });

    // f90 derived from F0 (Frostbite, Lagarde 2014) is what makes `dielectric = 0` mean literally no
    // specular while glTF's 0.04 still reaches full grazing reflectance: the 50/3 coefficient saturates
    // any F0 at or above ~0.02. A wrong coefficient leaves a spec-standard dielectric dim at grazing.
    test("the Fresnel f90 is derived from F0, so zero reflectance stays zero at every angle", () => {
        // vdh = 0 is grazing (the pow term is 1, so F is exactly f90)
        expect(fresnelSchlick(0, d.vec3f(0.04)).x).toBe(1);
        // 0.02 is the knee itself (0.06 · 50/3 = 1), so the f32 coefficient lands a hair under
        expect(fresnelSchlick(0, d.vec3f(0.02)).x).toBeCloseTo(1, 6);
        expect(fresnelSchlick(0, d.vec3f(0)).x).toBe(0); // a true zero-reflectance material
        expect(fresnelSchlick(0, d.vec3f(0.01)).x).toBeCloseTo(0.5, 5); // below the saturation knee
        // vdh = 1 is normal incidence: F is F0 itself, whatever f90 is
        expect(fresnelSchlick(1, d.vec3f(0.04)).x).toBeCloseTo(0.04, 6);
    });

    test("a rough dielectric reflects at grazing and stays dark on a back-facing light", () => {
        const s = Pbr({
            albedo: d.vec3f(0, 0, 0), // no diffuse, so what is left is the specular term alone
            metallic: 0,
            roughness: 0.5,
            occlusion: 1,
            dielectric: 0.04,
        });
        const N = d.vec3f(0, 1, 0);
        // a grazing view of a light near the mirror direction: the specular lobe is live
        const graze = brdf(s, N, d.vec3f(0.99, 0.14, 0), d.vec3f(-0.99, 0.14, 0));
        expect(graze.x).toBeGreaterThan(0.01);
        // a back-facing light: the clamped physical cosine zeroes it, however live the lobe is. Not exactly
        // antiparallel to V — `normalize(V + L)` is 0/0 there, which the CPU arm rejects outright (WGSL's
        // finite-math assumption); gpu.md's NaN policy says compute through, so the shader keeps the form
        expect(brdf(s, N, d.vec3f(0, 1, 0), d.vec3f(0.312, -0.95, 0)).x).toBe(0);
    });

    test("a view along the mirror direction of a metal catches the whole highlight", () => {
        const metal = Pbr({
            albedo: d.vec3f(1, 1, 1),
            metallic: 1,
            roughness: 0.1,
            occlusion: 1,
            dielectric: 0.04,
        });
        const N = d.vec3f(0, 1, 0);
        const L = d.vec3f(0, 1, 0);
        // the mirror direction of L about N is L itself, so a view along L catches the whole highlight
        const spec = brdf(metal, N, d.vec3f(0, 1, 0), L);
        expect(spec.x).toBeGreaterThan(1); // a tight GGX lobe concentrates far above the diffuse level
        // a metal carries no diffuse at all (kd scales by 1 − metallic), so the whole response is the GGX
        // lobe: roughening it spreads that energy and the peak collapses
        const rough = Pbr({ ...metal, roughness: 1 });
        expect(brdf(rough, N, d.vec3f(0, 1, 0), L).x).toBeLessThan(spec.x / 100);
    });

    // the sphere-source lobe is documented to reduce to the punctual one at radius 0: the representative
    // point collapses onto the light centre, aPrime = a, and the energy renormalization is 1
    test("brdfSphere at radius 0 is brdf", () => {
        const s = Pbr({
            albedo: d.vec3f(0.6, 0.5, 0.4),
            metallic: 0.3,
            roughness: 0.4,
            occlusion: 1,
            dielectric: 0.04,
        });
        const N = d.vec3f(0, 1, 0);
        // axis-aligned unit vectors, so `normalize(Lc * dist)` is exact and the two paths coincide bitwise
        for (const V of [d.vec3f(0, 1, 0), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1)]) {
            for (const dist of [1, 4]) {
                const a = brdf(s, N, V, d.vec3f(0, 1, 0));
                const b = brdfSphere(s, N, V, d.vec3f(0, 1, 0), dist, 0);
                expect(b.x).toBe(a.x);
                expect(b.y).toBe(a.y);
                expect(b.z).toBe(a.z);
            }
        }
    });

    test("a source radius spreads the highlight without brightening it", () => {
        const s = Pbr({
            albedo: d.vec3f(1, 1, 1),
            metallic: 1,
            roughness: 0.1,
            occlusion: 1,
            dielectric: 0.04,
        });
        const N = d.vec3f(0, 1, 0);
        const V = d.vec3f(0, 1, 0);
        const peak = brdfSphere(s, N, V, d.vec3f(0, 1, 0), 4, 0).x;
        const spread = brdfSphere(s, N, V, d.vec3f(0, 1, 0), 4, 1).x;
        expect(spread).toBeLessThan(peak); // the peak drops as the lobe widens (energy conserved)
        // off-peak, the wider lobe catches light the pinpoint one misses
        const off = d.vec3f(0.5, 0.866, 0);
        expect(brdfSphere(s, N, V, off, 4, 1).x).toBeGreaterThan(brdfSphere(s, N, V, off, 4, 0).x);
    });
});

describe("cascade selection", () => {
    // the receiver picks its cascade by camera-forward distance from the eye — the same axis the CPU
    // cascade fit splits along, which is why it is one function rather than two spellings
    test("viewDepth is the camera-forward distance from the eye", () => {
        const right = d.vec3f(1, 0, 0);
        const up = d.vec3f(0, 1, 0);
        const eye = d.vec3f(0, 2, 5);
        // forward = -cross(right, up) = -(0, 0, 1)·… → -z, so a point deeper into the scene is positive
        expect(viewDepth(right, up, eye, d.vec3f(0, 2, 0))).toBeCloseTo(5, 6);
        expect(viewDepth(right, up, eye, d.vec3f(0, 2, 6))).toBeCloseTo(-1, 6); // behind the eye
        // lateral offset doesn't change the depth
        expect(viewDepth(right, up, eye, d.vec3f(9, -3, 0))).toBeCloseTo(5, 6);
    });
});

describe("shadow uniform layouts", () => {
    // the staging writers index these by hand, so the schema and the offsets they use must agree — and
    // the values must not move silently across a repack (they are the shipped binding sizes)
    test("the sun params layout is the shipped one, derived from the schema", () => {
        expect(CASCADE_FLOATS).toBe(24); // mat4 16 + rect 4 + far + texelWorld + 2 pad
        expect(SHADOW_PARAMS_BYTES).toBe(MAX_CASCADES * 96 + 32);
        expect(SHADOW_PARAMS_BYTES).toBe(TAIL * 4 + 32);
    });

    // the staging writer indexes a Float32Array by these, so a reordered or resized struct must move them
    // with it — a stale hard-coded offset is the layout-mismatch bug the schemas exist to make impossible
    test("the params field indices are the ones the shipped WGSL struct reads", () => {
        expect(SUN_PARAMS.cascade.viewProj).toBe(0);
        expect(SUN_PARAMS.cascade.rect).toBe(16);
        expect(SUN_PARAMS.cascade.far).toBe(20);
        expect(SUN_PARAMS.cascade.texelWorld).toBe(21);
        // the tail is indexed absolutely (it follows the cascade array), not per-row
        expect(SUN_PARAMS.globals.count).toBe(TAIL);
        expect(SUN_PARAMS.globals.overlap).toBe(TAIL + 1);
        expect(SUN_PARAMS.globals.depthBias).toBe(TAIL + 2);
        expect(SUN_PARAMS.globals.enabled).toBe(TAIL + 3);
        expect(SUN_PARAMS.globals.normalBias).toBe(TAIL + 4);
        expect(SUN_PARAMS.globals.texel).toBe(TAIL + 5);
    });

    test("the caster stride is the shipped one", () => {
        expect(POINT_CASTER_FLOATS).toBe(20); // pos + nf + spotA/B/C
    });
});

describe("splice chunks", () => {
    // every consumer splices several of these into one module, so a shared dependency emitted by two of
    // them is a duplicate WGSL definition — the failure the shared namespace + base-forcing prevent
    const chunks = {
        pbr: pbrWgsl,
        caster: casterWgsl,
        pointShadow: pointShadowWgsl,
        sunStruct: sunStructWgsl,
        sunShadow: sunShadowWgsl,
    };

    const defs = (wgsl: string) =>
        [...wgsl.matchAll(/^(?:fn|struct)\s+([A-Za-z0-9_]+)/gm)].map((m) => m[1]);

    test("no chunk emits an anonymous or suffixed definition", () => {
        for (const [name, chunk] of Object.entries(chunks)) {
            for (const def of defs(chunk())) {
                expect(def, `${name} emits ${def}`).not.toMatch(/^item/);
                expect(def, `${name} emits ${def}`).not.toMatch(/_\d+$/);
            }
        }
    });

    test("every pair of chunks stays duplicate-free", () => {
        const entries = Object.entries(chunks);
        for (const [an, a] of entries) {
            for (const [bn, b] of entries) {
                if (an >= bn) continue;
                const all = [...defs(a()), ...defs(b())];
                const dupes = all.filter((x, i) => all.indexOf(x) !== i);
                expect(dupes, `${an} + ${bn}`).toEqual([]);
            }
        }
    });

    test("the relocatable receivers keep the names a raw splice site calls", () => {
        expect(pointShadowWgsl()).toContain(
            "fn pointShadowOf(light: PointLightGpu, normal: vec3f, fragWorld: vec3f) -> f32 {",
        );
        expect(sunShadowWgsl()).toContain(
            "fn sampleSunShadow(worldPos: vec3f, normal: vec3f) -> f32 {",
        );
        expect(casterWgsl()).toContain("struct PointCasters {");
        expect(casterWgsl()).toContain("struct TileRects {");
        expect(sunStructWgsl()).toContain("struct SunShadow {");
    });
});
