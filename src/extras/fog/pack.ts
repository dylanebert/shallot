import type { World } from "../../engine";
import { unpackColorInto } from "../../engine";
import { Fog } from "./index";
import { FOG_PARAMS } from "./medium";

/** Packs extinction density, scattering albedo and light intensity into caller-owned FogGpu storage.
 * A zero-extinction medium has zero in-scatter. */
export function packFog(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    unpackColorInto(world.storage(Fog).color.get(eid), out, FOG_PARAMS.color);
    const fog = world.storage(Fog);
    const extinction = fog.absorption.get(eid) + fog.scattering.get(eid);
    out[FOG_PARAMS.medium] = fog.density.get(eid) * extinction;
    out[FOG_PARAMS.medium + 1] = fog.heightBase.get(eid);
    out[FOG_PARAMS.medium + 2] = fog.heightFalloff.get(eid);
    out[FOG_PARAMS.medium + 3] = fog.viewDistance.get(eid);
    out[FOG_PARAMS.extra + 1] = fog.scatteringAsymmetry.get(eid);
    out[FOG_PARAMS.extra + 2] = extinction > 0 ? fog.scattering.get(eid) / extinction : 0;
    out[FOG_PARAMS.extra + 3] = fog.lightIntensity.get(eid);
}
