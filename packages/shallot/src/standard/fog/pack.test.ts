import { describe, expect, test } from "bun:test";
import { srgbToLinear } from "../../engine/utils/color";
import { Fog } from "./index";
import { FOG_MAX_STEPS, FOG_PARAMS } from "./march";
import { packFog } from "./pack";

// packFog is the CPU→GPU uniform boundary: it reads the `Fog` singleton's sparse components and writes the
// `FogGpu` staging float array the compute pipeline binds verbatim. `coding.md` names this crossing as
// earning a test outright — a wrong lane or a missed clamp here is invisible to every WGSL-side check,
// since the shader only ever sees the packed floats, never the component reads that produced them.
describe("packFog", () => {
    test("writes color (sRGB→linear), march, and extra at the schema's own offsets", () => {
        const eid = 101;
        Fog.color.set(eid, 0xff8040);
        Fog.density.set(eid, 0.05);
        Fog.heightBase.set(eid, 1);
        Fog.heightFalloff.set(eid, 0.2);
        Fog.jitter.set(eid, 0.5);
        Fog.steps.set(eid, 40);
        Fog.anisotropy.set(eid, 0.3);
        Fog.absorption.set(eid, 0.1);
        Fog.scattering.set(eid, 0.8);
        Fog.scatterIntensity.set(eid, 2);

        const out = new Float32Array(12);
        packFog(eid, out);

        expect(out[FOG_PARAMS.color]).toBeCloseTo(srgbToLinear(0xff / 255), 6);
        expect(out[FOG_PARAMS.color + 1]).toBeCloseTo(srgbToLinear(0x80 / 255), 6);
        expect(out[FOG_PARAMS.color + 2]).toBeCloseTo(srgbToLinear(0x40 / 255), 6);
        expect(out[FOG_PARAMS.march]).toBeCloseTo(0.05, 6);
        expect(out[FOG_PARAMS.march + 1]).toBe(1);
        expect(out[FOG_PARAMS.march + 2]).toBeCloseTo(0.2, 6);
        expect(out[FOG_PARAMS.march + 3]).toBe(0.5);
        expect(out[FOG_PARAMS.extra]).toBe(40);
        expect(out[FOG_PARAMS.extra + 1]).toBeCloseTo(0.3, 6);
        expect(out[FOG_PARAMS.extra + 2]).toBeCloseTo(0.1, 6);
        // `extra + 3` is `gain = scattering · scatterIntensity`, not a fourth raw field — the one lane
        // that isn't a direct component read
        expect(out[FOG_PARAMS.extra + 3]).toBeCloseTo(0.8 * 2, 6);
    });

    test("clamps steps to [1, FOG_MAX_STEPS] rather than passing an under- or over-range count through", () => {
        const eidLow = 102;
        Fog.steps.set(eidLow, 0);
        const outLow = new Float32Array(12);
        packFog(eidLow, outLow);
        expect(outLow[FOG_PARAMS.extra]).toBe(1);

        const eidHigh = 103;
        Fog.steps.set(eidHigh, 999);
        const outHigh = new Float32Array(12);
        packFog(eidHigh, outHigh);
        expect(outHigh[FOG_PARAMS.extra]).toBe(FOG_MAX_STEPS);
    });

    test("zeroes the staging array first, so an unwritten lane (color.a) reads 0, not stale data", () => {
        const eid = 104;
        const out = new Float32Array(12).fill(999);
        packFog(eid, out);
        expect(out[FOG_PARAMS.color + 3]).toBe(0);
    });
});
