import { unpackColor } from "../../engine";
import { Fog } from "./index";
import { FOG_MAX_STEPS, FOG_PARAMS } from "./march";

/** pack a `Fog` singleton entity into its uniform (the `FogGpu` schema's layout, `./march`). `steps` clamps
 * to `[1, FOG_MAX_STEPS]` so the GPU loop integrates the full ray at the cap resolution. `extra` carries the
 * scattering knobs for the S2 in-scatter march: `(steps, anisotropy g, absorption, gain)`, where
 * `gain = scattering · scatterIntensity` is the combined light-shaft brightness. */
export function packFog(eid: number, out: Float32Array): void {
    out.fill(0);
    const rgb = unpackColor(Fog.color.get(eid));
    out[FOG_PARAMS.color] = rgb.r;
    out[FOG_PARAMS.color + 1] = rgb.g;
    out[FOG_PARAMS.color + 2] = rgb.b;
    out[FOG_PARAMS.march] = Fog.density.get(eid);
    out[FOG_PARAMS.march + 1] = Fog.heightBase.get(eid);
    out[FOG_PARAMS.march + 2] = Fog.heightFalloff.get(eid);
    out[FOG_PARAMS.march + 3] = Fog.jitter.get(eid);
    out[FOG_PARAMS.extra] = Math.min(Math.max(Fog.steps.get(eid), 1), FOG_MAX_STEPS);
    out[FOG_PARAMS.extra + 1] = Fog.anisotropy.get(eid);
    out[FOG_PARAMS.extra + 2] = Fog.absorption.get(eid);
    out[FOG_PARAMS.extra + 3] = Fog.scattering.get(eid) * Fog.scatterIntensity.get(eid);
}
