import type { World } from "../../engine";
import { unpackColorInto } from "../../engine";
import { Sky } from "./index";
import { SKY_AT } from "./shader";

/**
 * Pack a `Sky` singleton entity into its {@link SkyGpu} uniform. Hex colors decode to linear RGB; per-light
 * SunDisk data comes from standard's lighting uniform.
 */
export function packSky(world: World, eid: number, out: Float32Array): void {
    out.fill(0);
    out[SKY_AT.brightness] = world.storage(Sky).brightness.get(eid);
    out[SKY_AT.hazeDensity] = world.storage(Sky).hazeDensity.get(eid);
    out[SKY_AT.horizonBand] = world.storage(Sky).band.get(eid);

    unpackColorInto(world.storage(Sky).hazeColor.get(eid), out, SKY_AT.hazeColor);

    unpackColorInto(world.storage(Sky).zenith.get(eid), out, SKY_AT.skyZenith);

    unpackColorInto(world.storage(Sky).horizon.get(eid), out, SKY_AT.skyHorizon);

    out[SKY_AT.starParams] = world.storage(Sky).starIntensity.get(eid);
    out[SKY_AT.starParams + 1] = world.storage(Sky).starAmount.get(eid);

    out[SKY_AT.cloudParams] = world.storage(Sky).cloudCoverage.get(eid);
    out[SKY_AT.cloudParams + 1] = world.storage(Sky).cloudDensity.get(eid);
    out[SKY_AT.cloudParams + 2] = world.storage(Sky).cloudHeight.get(eid);

    unpackColorInto(world.storage(Sky).cloudColor.get(eid), out, SKY_AT.cloudColor);
}
