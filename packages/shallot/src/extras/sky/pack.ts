import { unpackColor } from "../../engine";
import { Sky } from "./index";

/**
 * pack a `Sky` singleton entity into its uniform (the layout `./shader`'s WGSL `Sky` struct declares). Hex
 * colors decode to linear rgb. The sun *direction* is not packed; the shader reads it from sear's
 * `lighting` uniform.
 */
export function packSky(eid: number, out: Float32Array): void {
    out.fill(0);
    out[0] = Sky.hazeDensity.get(eid);
    out[1] = Sky.band.get(eid);

    const haze = unpackColor(Sky.hazeColor.get(eid));
    out[4] = haze.r;
    out[5] = haze.g;
    out[6] = haze.b;

    const zenith = unpackColor(Sky.zenith.get(eid));
    out[8] = zenith.r;
    out[9] = zenith.g;
    out[10] = zenith.b;

    const horizon = unpackColor(Sky.horizon.get(eid));
    out[12] = horizon.r;
    out[13] = horizon.g;
    out[14] = horizon.b;

    out[16] = Sky.starIntensity.get(eid);
    out[17] = Sky.starAmount.get(eid);

    out[20] = Sky.cloudCoverage.get(eid);
    out[21] = Sky.cloudDensity.get(eid);
    out[22] = Sky.cloudHeight.get(eid);

    const cloud = unpackColor(Sky.cloudColor.get(eid));
    out[24] = cloud.r;
    out[25] = cloud.g;
    out[26] = cloud.b;

    out[28] = Sky.sunSize.get(eid);
    out[31] = Sky.sunGlow.get(eid);

    const sun = unpackColor(Sky.sunColor.get(eid));
    out[32] = sun.r;
    out[33] = sun.g;
    out[34] = sun.b;
}
