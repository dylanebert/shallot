import { traits, type State } from "../../engine";
import { formatHex } from "../../engine/ecs/core";
import { Transform, WorldTransform } from "../transforms";
import { unpackColor, normalizeDirection } from "../../engine";

export const AmbientLight = {
    color: [] as number[],
    intensity: [] as number[],
};

traits(AmbientLight, {
    defaults: () => ({ color: 0xffffff, intensity: 0.5 }),
    format: { color: formatHex },
});

export const PointLight = {
    color: [] as number[],
    intensity: [] as number[],
    radius: [] as number[],
    shadows: [] as number[],
};

traits(PointLight, {
    requires: [Transform],
    defaults: () => ({ color: 0xffffff, intensity: 1.0, radius: 10, shadows: 0 }),
    format: { color: formatHex },
});

export const DirectionalLight = {
    color: [] as number[],
    intensity: [] as number[],
    directionX: [] as number[],
    directionY: [] as number[],
    directionZ: [] as number[],
    shadows: [] as number[],
};

traits(DirectionalLight, {
    defaults: () => ({
        color: 0xffffff,
        intensity: 1.5,
        directionX: -0.6,
        directionY: -1.0,
        directionZ: -0.8,
        shadows: 1,
    }),
    format: { color: formatHex },
});

interface AmbientLightData {
    color: number;
    intensity: number;
}

interface DirectionalLightData {
    color: number;
    intensity: number;
    directionX: number;
    directionY: number;
    directionZ: number;
}

const lightData = new Float32Array(12);

export function packLightUniforms(
    ambient: AmbientLightData,
    directional: DirectionalLightData,
): Float32Array {
    const ambientRgb = unpackColor(ambient.color);
    lightData[0] = ambientRgb.r;
    lightData[1] = ambientRgb.g;
    lightData[2] = ambientRgb.b;
    lightData[3] = ambient.intensity;

    const [dx, dy, dz] = normalizeDirection(
        directional.directionX,
        directional.directionY,
        directional.directionZ,
    );
    lightData[4] = dx;
    lightData[5] = dy;
    lightData[6] = dz;
    lightData[7] = 0;

    const sunRgb = unpackColor(directional.color);
    lightData[8] = sunRgb.r * directional.intensity;
    lightData[9] = sunRgb.g * directional.intensity;
    lightData[10] = sunRgb.b * directional.intensity;
    lightData[11] = 0;

    return lightData;
}

export const MAX_RASTER_POINT_LIGHTS = 64;
export const POINT_LIGHT_STRIDE = 8;
export const POINT_LIGHT_BUFFER_SIZE = MAX_RASTER_POINT_LIGHTS * POINT_LIGHT_STRIDE * 4;

const pointLightScratch = new Float32Array(MAX_RASTER_POINT_LIGHTS * POINT_LIGHT_STRIDE);

export function packPointLights(state: State, cameraShadows: boolean): [Float32Array, number] {
    let count = 0;
    let shadowCount = 0;
    let overflow = false;
    for (const eid of state.query([PointLight])) {
        if (count >= MAX_RASTER_POINT_LIGHTS) {
            overflow = true;
            break;
        }
        const o = count * POINT_LIGHT_STRIDE;
        const world = WorldTransform.data;
        pointLightScratch[o] = world[eid * 16 + 12];
        pointLightScratch[o + 1] = world[eid * 16 + 13];
        pointLightScratch[o + 2] = world[eid * 16 + 14];
        pointLightScratch[o + 3] = PointLight.radius[eid];
        const rgb = unpackColor(PointLight.color[eid]);
        const intensity = PointLight.intensity[eid];
        pointLightScratch[o + 4] = rgb.r * intensity;
        pointLightScratch[o + 5] = rgb.g * intensity;
        pointLightScratch[o + 6] = rgb.b * intensity;
        if (cameraShadows && PointLight.shadows[eid] !== 0) {
            pointLightScratch[o + 7] = shadowCount;
            shadowCount++;
        } else {
            pointLightScratch[o + 7] = -1;
        }
        count++;
    }
    if (overflow) console.warn(`point light cap reached (${MAX_RASTER_POINT_LIGHTS})`);
    return [pointLightScratch, count];
}
