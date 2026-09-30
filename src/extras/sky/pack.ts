import type { World } from "../../engine";
import { unpackColor } from "../../engine";
import { Sky } from "./index";
import { SKY_AT } from "./shader";

/**
 * pack a `Sky` singleton entity into its {@link SkyGpu} uniform. Hex colors decode to linear rgb. The sun
 * *direction* is not packed; the shader reads it from sear's `lighting` uniform.
 */
export function packSky(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    out[SKY_AT.hazeDensity] = world.storage(Sky).hazeDensity.get(eid);
    out[SKY_AT.horizonBand] = world.storage(Sky).band.get(eid);

    const haze = unpackColor(world.storage(Sky).hazeColor.get(eid));
    out[SKY_AT.hazeColor] = haze.r;
    out[SKY_AT.hazeColor + 1] = haze.g;
    out[SKY_AT.hazeColor + 2] = haze.b;

    const zenith = unpackColor(world.storage(Sky).zenith.get(eid));
    out[SKY_AT.skyZenith] = zenith.r;
    out[SKY_AT.skyZenith + 1] = zenith.g;
    out[SKY_AT.skyZenith + 2] = zenith.b;

    const horizon = unpackColor(world.storage(Sky).horizon.get(eid));
    out[SKY_AT.skyHorizon] = horizon.r;
    out[SKY_AT.skyHorizon + 1] = horizon.g;
    out[SKY_AT.skyHorizon + 2] = horizon.b;

    out[SKY_AT.starParams] = world.storage(Sky).starIntensity.get(eid);
    out[SKY_AT.starParams + 1] = world.storage(Sky).starAmount.get(eid);

    out[SKY_AT.cloudParams] = world.storage(Sky).cloudCoverage.get(eid);
    out[SKY_AT.cloudParams + 1] = world.storage(Sky).cloudDensity.get(eid);
    out[SKY_AT.cloudParams + 2] = world.storage(Sky).cloudHeight.get(eid);

    const cloud = unpackColor(world.storage(Sky).cloudColor.get(eid));
    out[SKY_AT.cloudColor] = cloud.r;
    out[SKY_AT.cloudColor + 1] = cloud.g;
    out[SKY_AT.cloudColor + 2] = cloud.b;

    out[SKY_AT.sunParams] = world.storage(Sky).sunSize.get(eid);
    out[SKY_AT.sunParams + 3] = world.storage(Sky).sunGlow.get(eid);

    const sun = unpackColor(world.storage(Sky).sunColor.get(eid));
    out[SKY_AT.sunVisualColor] = sun.r;
    out[SKY_AT.sunVisualColor + 1] = sun.g;
    out[SKY_AT.sunVisualColor + 2] = sun.b;
}
