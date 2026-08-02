import { beforeEach, describe, expect, spyOn, test } from "bun:test";
import * as d from "typegpu/data";
import { bitcastU32toF32 } from "typegpu/std";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { octEncodeNormal } from "../../engine/utils/core";
import { Slab } from "../slab";
import { Transform } from "../transforms";
import {
    distanceAttenuation,
    LIGHTING_UNIFORM_SIZE,
    LightingGpu,
    MAX_POINT_LIGHTS,
    PointLight,
    PointLightGpu,
    PointLights,
    PointLightsRw,
    pointLightsWgsl,
    spotFactor,
    spotParams,
    warnLightOverflow,
} from "./lighting";

// Pure-CPU truth for point lights: the falloff oracle sear's WGSL twin is pinned to, and the
// list-cap warn. The list itself is GPU-built (cluster.ts); the gym `render` scenario gates it.

describe("distanceAttenuation", () => {
    // radiusSq is the source-sphere clamp `max(d², radius²)`; 1e-4 = (0.01 m)² is the old punctual
    // filament, so passing it reproduces the pre-radius behavior these three cases pin
    const Punctual = 1e-4;

    test("inverse-square in the near field", () => {
        // d = 0.5 against range 100: the window term deviates from 1 by (d²/r²)² ≈ 6e-10,
        // far below the assertion precision, so the value is 1/d²
        const r = 100;
        expect(distanceAttenuation(0.25, 1 / (r * r), Punctual)).toBeCloseTo(1 / 0.25, 5);
    });

    test("exactly zero at the range and past it", () => {
        const invRangeSq = 1 / 16; // range 4
        expect(distanceAttenuation(16, invRangeSq, Punctual)).toBe(0);
        expect(distanceAttenuation(100, invRangeSq, Punctual)).toBe(0);
    });

    test("monotonically nonincreasing out to the range", () => {
        const invRangeSq = 1 / 25;
        let prev = Number.POSITIVE_INFINITY;
        for (let d = 0.1; d <= 5; d += 0.1) {
            const a = distanceAttenuation(d * d, invRangeSq, Punctual);
            expect(a).toBeLessThanOrEqual(prev);
            prev = a;
        }
    });

    test("plateaus inside the source radius instead of spiking", () => {
        // a 0.1 m sphere source against a far range (window ≈ 1 throughout): inside the radius the
        // `max(d², radius²)` clamp pins the denominator at radius², so the intensity is flat at
        // `1/radius²` from the center out to the surface, rather than the punctual 1/d² spike toward ∞
        const radius = 0.1;
        const radiusSq = radius * radius; // 0.01
        const invRangeSq = 1 / (100 * 100);
        const center = distanceAttenuation(0, invRangeSq, radiusSq);
        const halfIn = distanceAttenuation((radius / 2) * (radius / 2), invRangeSq, radiusSq);
        const atSurface = distanceAttenuation(radiusSq, invRangeSq, radiusSq);
        // all three sit at the plateau 1/radius² = 100 (window deviation < 1e-7 at this range)
        expect(center).toBeCloseTo(100, 4);
        expect(halfIn).toBeCloseTo(100, 4);
        expect(atSurface).toBeCloseTo(100, 4);
        // just outside the radius the inverse-square has taken over and dropped below the plateau
        const justOut = distanceAttenuation(radiusSq * 4, invRangeSq, radiusSq); // d = 2·radius
        expect(justOut).toBeLessThan(center);
        expect(justOut).toBeCloseTo(25, 3); // 1/(2·radius)²
    });
});

describe("spotParams", () => {
    // the GPU compact pass derives a spot's (scale, offset) from its inner/outer cone half-angles (degrees);
    // the FS angular factor is `saturate(cd·scale + offset)²`, cd = cos(angle between the cone axis and the
    // light→fragment direction). This pins the derivation to its defining boundary: full at the inner edge,
    // zero at the outer — sear's WGSL twin matches it.
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const angular = (cd: number, p: { scale: number; offset: number }) => {
        const a = Math.min(Math.max(cd * p.scale + p.offset, 0), 1);
        return a * a;
    };

    test("full at the inner cone edge, zero at the outer", () => {
        const p = spotParams(20, 30);
        expect(angular(Math.cos(rad(20)), p)).toBeCloseTo(1, 6);
        expect(angular(Math.cos(rad(30)), p)).toBeCloseTo(0, 6);
        expect(angular(1, p)).toBe(1); // dead on axis — saturated full
        expect(angular(Math.cos(rad(40)), p)).toBe(0); // outside the outer cone — dark
    });

    test("monotonic across the penumbra (brighter toward the axis)", () => {
        const p = spotParams(10, 40);
        let prev = -1;
        for (let deg = 40; deg >= 10; deg -= 2) {
            const a = angular(Math.cos(rad(deg)), p);
            expect(a).toBeGreaterThanOrEqual(prev);
            prev = a;
        }
    });

    test("degenerate inner == outer is a hard edge, not a divide by zero", () => {
        const p = spotParams(25, 25);
        expect(Number.isFinite(p.scale)).toBe(true);
        expect(angular(Math.cos(rad(24)), p)).toBe(1); // just inside — full
        expect(angular(Math.cos(rad(26)), p)).toBe(0); // just outside — dark
    });
});

describe("warnLightOverflow", () => {
    let state: State;

    beforeEach(() => {
        clear();
        register("Transform", Transform);
        register("PointLight", PointLight);
        Slab.collect(); // allocate the slab-backed fields CPU-side
        state = new State();
    });

    function light(i: number) {
        const eid = state.create();
        state.add(eid, PointLight);
        state.add(eid, Transform);
        Transform.pos.set(eid, i, 0, 0, 0);
        return eid;
    }

    test("warns once per episode past the cap, re-arms under it", () => {
        const warn = spyOn(console, "warn");
        const eids: number[] = [];
        for (let i = 0; i < MAX_POINT_LIGHTS + 3; i++) eids.push(light(i));
        warnLightOverflow(state);
        warnLightOverflow(state);
        expect(warn).toHaveBeenCalledTimes(1); // latched within the episode
        for (const eid of eids.slice(MAX_POINT_LIGHTS - 4)) state.destroy(eid);
        warnLightOverflow(state); // under the cap — re-arms
        for (let i = 0; i < 8; i++) light(1000 + i);
        warnLightOverflow(state);
        expect(warn).toHaveBeenCalledTimes(2); // a new episode warns again
        warn.mockRestore();
    });
});

// `LightingGpu` is the one schema both a typed pipeline's layout entry (the fog march's `lighting`
// binding) and sear's own splice (`lightingWgsl()`) read — a field reorder with no matching edit on the
// CPU staging writer compiles fine on both sides and reads garbage across the boundary, so the byte size
// is pinned here rather than left to the bench.
describe("LightingGpu schema", () => {
    test("matches LIGHTING_UNIFORM_SIZE — the CPU staging array's byte length", () => {
        expect(d.sizeOf(LightingGpu)).toBe(LIGHTING_UNIFORM_SIZE);
    });
});

// The compacted list's schema is the CPU↔GPU boundary two raw-WGSL splice sites (sear's clustered loop,
// the fog march) read by field name, and the compact pass writes through a second, atomic-count schema.
// Both facts are silent-failure shaped — a renamed field or a diverged layout compiles fine and reads
// garbage — so they're pinned here rather than left to the bench.
describe("PointLights schema", () => {
    test("the reader and the compact pass's writer schemas have identical layouts", () => {
        expect(d.sizeOf(PointLightsRw)).toBe(d.sizeOf(PointLights));
        expect(d.alignmentOf(PointLightsRw)).toBe(d.alignmentOf(PointLights));
        // the count header must occupy the same 16 B in both, so `lights` starts at the same offset
        expect(d.sizeOf(PointLightsRw.propTypes.count)).toBe(d.sizeOf(PointLights.propTypes.count));
        expect(d.sizeOf(PointLightsRw.propTypes.lights)).toBe(
            d.sizeOf(PointLights.propTypes.lights),
        );
    });

    test("the spliced WGSL keeps the names a raw consumer indexes by", () => {
        const wgsl = pointLightsWgsl();
        expect(wgsl).toContain("struct PointLightGpu");
        expect(wgsl).toContain("struct PointLights");
        expect(wgsl).toMatch(/count: vec4u/);
        expect(wgsl).toMatch(new RegExp(`lights: array<PointLightGpu, ${MAX_POINT_LIGHTS}>`));
        for (const field of ["posRange", "color", "params"]) {
            expect(wgsl).toContain(`${field}: vec4f`);
        }
    });
});

// `spotFactor` is the cone attenuation the shader multiplies in, and the same function on the CPU — so the
// (scale, offset) pair `spotParams` bakes into the compacted light can be checked end to end here
describe("spotFactor", () => {
    const light = (axis: [number, number, number], inner: number, outer: number) => {
        const p = spotParams(inner, outer);
        return PointLightGpu({
            posRange: d.vec4f(0, 0, 0, 1),
            color: d.vec4f(1, 1, 1, 0),
            // the compact pass stores the cone axis oct-packed, bitcast into the params.y float lane
            params: d.vec4f(
                1,
                bitcastU32toF32(octEncodeNormal(d.vec3f(...axis))),
                p.scale,
                p.offset,
            ),
        });
    };
    // L points fragment→light, so a fragment on the cone axis sees L = -axis
    const onAxis = d.vec3f(0, 1, 0);
    const axis: [number, number, number] = [0, -1, 0];

    test("a plain point light is a no-op", () => {
        const plain = PointLightGpu({
            posRange: d.vec4f(0, 0, 0, 1),
            color: d.vec4f(1, 1, 1, 0),
            params: d.vec4f(1, 0, 0, 1), // what the compact pass writes for a non-spot
        });
        expect(spotFactor(plain, onAxis)).toBe(1);
    });

    test("full brightness inside the inner cone, dark past the outer", () => {
        const l = light(axis, 20, 30);
        expect(spotFactor(l, onAxis)).toBe(1); // dead centre
        const at = (deg: number) => {
            const r = (deg * Math.PI) / 180;
            return spotFactor(l, d.vec3f(Math.sin(r), Math.cos(r), 0));
        };
        expect(at(19)).toBe(1); // inside the inner half-angle
        expect(at(31)).toBe(0); // past the outer half-angle
        expect(at(25)).toBeGreaterThan(0); // and monotone between them
        expect(at(25)).toBeLessThan(1);
        expect(at(28)).toBeLessThan(at(22));
        // squared, not the bare ramp: half-way across the cosine band reads 0.5² — the Frostbite form
        const half = Math.acos((Math.cos(Math.PI / 9) + Math.cos(Math.PI / 6)) / 2);
        expect(at((half * 180) / Math.PI)).toBeCloseTo(0.25, 5);
    });

    test("behind the cone is dark, never negative", () => {
        const l = light(axis, 20, 30);
        expect(spotFactor(l, d.vec3f(0, -1, 0))).toBe(0);
    });
});
