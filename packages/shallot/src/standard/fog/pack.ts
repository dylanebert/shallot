import { unpackColor } from "../../engine";
import { Fog } from "./index";
import { FOG_MAX_STEPS } from "./march";

/** pack a `Fog` singleton entity into its uniform (the layout `./march`'s WGSL `Fog` struct declares).
 * `steps` clamps to `[1, FOG_MAX_STEPS]` so the GPU loop integrates the full ray at the cap resolution.
 * `extra` carries the scattering knobs for the S2 in-scatter march: `(steps, anisotropy g, absorption,
 * gain)`, where `gain = scattering · scatterIntensity` is the combined light-shaft brightness. */
export function packFog(eid: number, out: Float32Array): void {
    out.fill(0);
    const rgb = unpackColor(Fog.color.get(eid));
    out[0] = rgb.r;
    out[1] = rgb.g;
    out[2] = rgb.b;
    out[4] = Fog.density.get(eid);
    out[5] = Fog.heightBase.get(eid);
    out[6] = Fog.heightFalloff.get(eid);
    out[7] = Fog.jitter.get(eid);
    out[8] = Math.min(Math.max(Fog.steps.get(eid), 1), FOG_MAX_STEPS);
    out[9] = Fog.anisotropy.get(eid);
    out[10] = Fog.absorption.get(eid);
    out[11] = Fog.scattering.get(eid) * Fog.scatterIntensity.get(eid);
}
