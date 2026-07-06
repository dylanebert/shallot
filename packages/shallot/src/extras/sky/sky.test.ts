import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../..";
import { clear, register } from "../../engine/ecs/core";
import { srgbToLinear } from "../../engine/utils/color";
import { Sky, SkyPlugin } from ".";
import { packSky } from "./pack";
import { SKY_FLOATS } from "./shader";

// packSky crosses the CPU→GPU boundary: it lays a `Sky` singleton into the std140 uniform `./shader`'s WGSL
// `Sky` struct declares. The contract is the field→offset map (a layout bug puts a value one slot off and
// the sky shades wrong) and the hex→linear decode. The real-GPU render lives in the gym `render` `sky` mode.
describe("packSky", () => {
    let state: State;

    beforeEach(() => {
        clear();
        state = new State();
        register("Sky", Sky, SkyPlugin.traits?.Sky);
    });

    test("lays scalars and hex-decoded colors into the std140 Sky offsets", () => {
        const eid = state.create();
        state.add(eid, Sky); // defaults applied, so every field is set
        Sky.hazeDensity.set(eid, 0.01);
        Sky.band.set(eid, 0.5);
        Sky.zenith.set(eid, 0xff8000);
        Sky.starIntensity.set(eid, 0.3);
        Sky.starAmount.set(eid, 0.6);
        Sky.cloudHeight.set(eid, 4);
        Sky.sunSize.set(eid, 0.7);
        Sky.sunGlow.set(eid, 0.2);
        Sky.sunColor.set(eid, 0x000000);

        const out = new Float32Array(SKY_FLOATS);
        packSky(eid, out);

        // leading f32 lanes
        expect(out[0]).toBeCloseTo(0.01, 6); // hazeDensity
        expect(out[1]).toBeCloseTo(0.5, 6); // band
        // skyZenith at the third vec4 (float 8): 0xff8000 → linear rgb, blue 0
        expect(out[8]).toBeCloseTo(srgbToLinear(1), 6);
        expect(out[9]).toBeCloseTo(srgbToLinear(0x80 / 255), 6);
        expect(out[10]).toBe(0);
        // starParams = (intensity, amount, _, _)
        expect(out[16]).toBeCloseTo(0.3, 6);
        expect(out[17]).toBeCloseTo(0.6, 6);
        // cloudParams = (coverage, density, height, _)
        expect(out[22]).toBe(4);
        // sunParams = (size, _, _, glow)
        expect(out[28]).toBeCloseTo(0.7, 6);
        expect(out[31]).toBeCloseTo(0.2, 6);
        // sunVisualColor at float 32: 0x000000 → 0
        expect(out[32]).toBe(0);
    });
});
