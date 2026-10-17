import { expect, test } from "bun:test";
import { World } from "../../engine";
import { Fog } from "./index";
import { FOG_FLOATS, FOG_PARAMS } from "./medium";
import { packFog } from "./pack";

test("fog packs coefficient extinction and scattering albedo independently of light intensity", () => {
    const world = new World();
    const eid = world.create();
    world.add(eid, Fog, {
        density: 0.25,
        absorption: 2,
        scattering: 6,
        lightIntensity: 3,
        viewDistance: 64,
    });
    const out = new Float32Array(FOG_FLOATS);
    packFog(world, eid, out);
    expect(out[FOG_PARAMS.medium]).toBe(2);
    expect(out[FOG_PARAMS.medium + 3]).toBe(64);
    world.storage(Fog).viewDistance.set(eid, 120);
    packFog(world, eid, out);
    expect(out[FOG_PARAMS.medium + 3]).toBe(120);
    expect(out[FOG_PARAMS.extra + 2]).toBe(0.75);
    expect(out[FOG_PARAMS.extra + 3]).toBe(3);
    world.storage(Fog).scattering.set(eid, 0);
    packFog(world, eid, out);
    expect(out[FOG_PARAMS.medium]).toBe(0.5);
    expect(out[FOG_PARAMS.extra + 2]).toBe(0);
    world.storage(Fog).absorption.set(eid, 0);
    packFog(world, eid, out);
    expect(out[FOG_PARAMS.medium]).toBe(0);
    expect(out[FOG_PARAMS.extra + 2]).toBe(0);
    world.dispose();
});
