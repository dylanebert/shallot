import type { World } from "../../engine";
import { unpackColorInto } from "../../engine";
import { Fog } from "./index";
import { FOG_MAX_STEPS, FOG_PARAMS } from "./march";

/** Packs into caller-owned storage using FogGpu's layout; clamps steps to the GPU loop's range.
 * Gain reverses the surface-light conversion for volumetric sources, outside the phase function. */
export function packFog(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    unpackColorInto(world.storage(Fog).color.get(eid), out, FOG_PARAMS.color);
    out[FOG_PARAMS.march] = world.storage(Fog).density.get(eid);
    out[FOG_PARAMS.march + 1] = world.storage(Fog).heightBase.get(eid);
    out[FOG_PARAMS.march + 2] = world.storage(Fog).heightFalloff.get(eid);
    out[FOG_PARAMS.march + 3] = world.storage(Fog).jitter.get(eid);
    out[FOG_PARAMS.extra] = Math.min(Math.max(world.storage(Fog).steps.get(eid), 1), FOG_MAX_STEPS);
    out[FOG_PARAMS.extra + 1] = world.storage(Fog).anisotropy.get(eid);
    out[FOG_PARAMS.extra + 2] = world.storage(Fog).absorption.get(eid);
    out[FOG_PARAMS.extra + 3] =
        (world.storage(Fog).scattering.get(eid) * world.storage(Fog).scatterIntensity.get(eid)) /
        Math.PI;
}
