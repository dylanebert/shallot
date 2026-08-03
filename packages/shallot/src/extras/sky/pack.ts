import { unpackColor } from "../../engine";
import { Sky } from "./index";
import { SKY_AT } from "./shader";

/**
 * pack a `Sky` singleton entity into its {@link SkyGpu} uniform. Hex colors decode to linear rgb. The sun
 * *direction* is not packed; the shader reads it from sear's `lighting` uniform.
 */
export function packSky(eid: number, out: Float32Array): void {
    out.fill(0);
    out[SKY_AT.hazeDensity] = Sky.hazeDensity.get(eid);
    out[SKY_AT.horizonBand] = Sky.band.get(eid);

    const haze = unpackColor(Sky.hazeColor.get(eid));
    out[SKY_AT.hazeColor] = haze.r;
    out[SKY_AT.hazeColor + 1] = haze.g;
    out[SKY_AT.hazeColor + 2] = haze.b;

    const zenith = unpackColor(Sky.zenith.get(eid));
    out[SKY_AT.skyZenith] = zenith.r;
    out[SKY_AT.skyZenith + 1] = zenith.g;
    out[SKY_AT.skyZenith + 2] = zenith.b;

    const horizon = unpackColor(Sky.horizon.get(eid));
    out[SKY_AT.skyHorizon] = horizon.r;
    out[SKY_AT.skyHorizon + 1] = horizon.g;
    out[SKY_AT.skyHorizon + 2] = horizon.b;

    out[SKY_AT.starParams] = Sky.starIntensity.get(eid);
    out[SKY_AT.starParams + 1] = Sky.starAmount.get(eid);

    out[SKY_AT.cloudParams] = Sky.cloudCoverage.get(eid);
    out[SKY_AT.cloudParams + 1] = Sky.cloudDensity.get(eid);
    out[SKY_AT.cloudParams + 2] = Sky.cloudHeight.get(eid);

    const cloud = unpackColor(Sky.cloudColor.get(eid));
    out[SKY_AT.cloudColor] = cloud.r;
    out[SKY_AT.cloudColor + 1] = cloud.g;
    out[SKY_AT.cloudColor + 2] = cloud.b;

    out[SKY_AT.sunParams] = Sky.sunSize.get(eid);
    out[SKY_AT.sunParams + 3] = Sky.sunGlow.get(eid);

    const sun = unpackColor(Sky.sunColor.get(eid));
    out[SKY_AT.sunVisualColor] = sun.r;
    out[SKY_AT.sunVisualColor + 1] = sun.g;
    out[SKY_AT.sunVisualColor + 2] = sun.b;
}
