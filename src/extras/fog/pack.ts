import type { World } from "../../engine";
import { unpackColorInto } from "../../engine";
import { Fog } from "./index";
import { FOG_MAX_STEPS, FOG_PARAMS } from "./march";

/** Packs extinction density, scattering albedo and light intensity into caller-owned FogGpu storage.
 * Clamps stepCount to the GPU loop's range; a zero-extinction medium has zero in-scatter. */
export function packFog(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    unpackColorInto(world.storage(Fog).color.get(eid), out, FOG_PARAMS.color);
    const fog = world.storage(Fog);
    const extinction = fog.absorption.get(eid) + fog.scattering.get(eid);
    out[FOG_PARAMS.march] = fog.density.get(eid) * extinction;
    out[FOG_PARAMS.march + 1] = world.storage(Fog).heightBase.get(eid);
    out[FOG_PARAMS.march + 2] = world.storage(Fog).heightFalloff.get(eid);
    out[FOG_PARAMS.march + 3] = world.storage(Fog).jitter.get(eid);
    out[FOG_PARAMS.extra] = Math.min(Math.max(fog.stepCount.get(eid), 1), FOG_MAX_STEPS);
    out[FOG_PARAMS.extra + 1] = fog.scatteringAsymmetry.get(eid);
    out[FOG_PARAMS.extra + 2] = extinction > 0 ? fog.scattering.get(eid) / extinction : 0;
    out[FOG_PARAMS.extra + 3] = fog.lightIntensity.get(eid);
}
