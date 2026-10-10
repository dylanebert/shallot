import type { World } from "../../engine";
import { unpackColor } from "../../engine";
import { Fog } from "./index";
import { FOG_MAX_STEPS, FOG_PARAMS } from "./march";

/** pack a `Fog` singleton entity into its uniform (the `FogGpu` schema's layout, `./march`). `steps` clamps
 * to `[1, FOG_MAX_STEPS]` so the GPU loop integrates the full ray at the cap resolution. `extra` carries the
 * scattering knobs for the S2 in-scatter march: `(steps, anisotropy g, absorption, gain)`. Gain includes a
 * `1/π` calibration that reverses the surface-light conversion for volumetric sources, preserving the former
 * shaft response without folding that factor into the Henyey–Greenstein phase. */
export function packFog(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    const rgb = unpackColor(world.storage(Fog).color.get(eid));
    out[FOG_PARAMS.color] = rgb.r;
    out[FOG_PARAMS.color + 1] = rgb.g;
    out[FOG_PARAMS.color + 2] = rgb.b;
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
