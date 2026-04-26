import { traits, relation, type State } from "../../engine";
import { formatHex } from "../../engine/ecs/core";
import {
    perspective,
    orthographic,
    multiply,
    invert,
    invertMatrix,
    unpackColor,
} from "../../engine";
import { Transform, WorldTransform } from "../transforms";
import { DirectionalLight } from "./light";
import { SCENE_UNIFORM_SIZE, SKY_UNIFORM_SIZE } from "./scene";

export const CameraMode = {
    Perspective: 0,
    Orthographic: 1,
} as const;

export const Camera = {
    fov: [] as number[],
    near: [] as number[],
    far: [] as number[],
    active: [] as number[],
    clearColor: [] as number[],
    mode: [] as number[],
    size: [] as number[],
};

traits(Camera, {
    requires: [Transform],
    defaults: () => ({
        fov: 60,
        near: 0.1,
        far: 1000,
        active: 1,
        clearColor: 0x0e0d0c,
        mode: CameraMode.Perspective,
        size: 5,
    }),
    format: { clearColor: formatHex },
    enums: { mode: CameraMode },
});

export const Tonemap = {
    exposure: [] as number[],
};

traits(Tonemap, {
    defaults: () => ({ exposure: 1.0 }),
});

export const FXAA = {};

export const Vignette = {
    strength: [] as number[],
    inner: [] as number[],
    outer: [] as number[],
};

traits(Vignette, {
    defaults: () => ({ strength: 0.5, inner: 0.4, outer: 0.8 }),
});

export const Posterize = {
    bands: [] as number[],
};

traits(Posterize, {
    defaults: () => ({ bands: 32 }),
});

export const Dither = {
    strength: [] as number[],
};

traits(Dither, {
    defaults: () => ({ strength: 0.1 }),
});

export const Shadows = {
    softness: [] as number[],
    samples: [] as number[],
    distance: [] as number[],
};

traits(Shadows, {
    defaults: () => ({ softness: 0, samples: 1, distance: 100 }),
});

export const Reflections = {};

export const Haze = {
    density: [] as number[],
    color: [] as number[],
};

traits(Haze, {
    defaults: () => ({
        density: 0.005,
        color: 0x4078d0,
    }),
    format: { color: formatHex },
});

export const Sky = {
    zenith: [] as number[],
    horizon: [] as number[],
    band: [] as number[],
};

traits(Sky, {
    defaults: () => ({
        zenith: 0x4078d0,
        horizon: 0x4098d8,
        band: 0,
    }),
    format: { zenith: formatHex, horizon: formatHex },
});

export const Moon = {
    phase: [] as number[],
    opacity: [] as number[],
    azimuth: [] as number[],
    elevation: [] as number[],
};

traits(Moon, {
    defaults: () => ({
        phase: 0.5,
        opacity: 1.0,
        azimuth: 45,
        elevation: 30,
    }),
});

export const Stars = {
    intensity: [] as number[],
    amount: [] as number[],
};

traits(Stars, {
    defaults: () => ({
        intensity: 0.8,
        amount: 0.5,
    }),
});

export const Clouds = {
    coverage: [] as number[],
    density: [] as number[],
    height: [] as number[],
    color: [] as number[],
};

traits(Clouds, {
    defaults: () => ({
        coverage: 0.7,
        density: 0.8,
        height: 0.5,
        color: 0xffffff,
    }),
    format: { color: formatHex },
});

export const Sun = {
    size: [] as number[],
    color: [] as number[],
    glow: [] as number[],
    azimuth: [] as number[],
    elevation: [] as number[],
};

traits(Sun, {
    defaults: () => ({
        size: 1.0,
        color: 0xfff0d8,
        glow: 0.3,
        azimuth: 0,
        elevation: 45,
    }),
    format: { color: formatHex },
});

export const Viewport = {
    width: [] as number[],
    height: [] as number[],
};

traits(Viewport, {
    defaults: () => ({ width: 0, height: 0 }),
});

export const RenderTarget = relation("render-target", { exclusive: true });

interface SkyParams {
    zenith: number;
    horizon: number;
    band: number;
}

interface HazeParams {
    density: number;
    color: number;
}

interface MoonParams {
    phase: number;
    opacity: number;
    azimuth: number;
    elevation: number;
}

interface StarsParams {
    intensity: number;
    amount: number;
}

interface CloudsParams {
    coverage: number;
    density: number;
    height: number;
    color: number;
}

interface SunParams {
    size: number;
    color: number;
    glow: number;
    azimuth: number;
    elevation: number;
}

const sceneBuffer = new ArrayBuffer(SCENE_UNIFORM_SIZE);
export const sceneStaging = new Float32Array(sceneBuffer);
export const sceneStagingU32 = new Uint32Array(sceneBuffer);

const _proj = new Float32Array(16);
const _view = new Float32Array(16);
const _ivp = new Float32Array(16);

export function uploadCamera(
    viewProj: Float32Array,
    eid: number,
    width: number,
    height: number,
    shadowSoftness: number = 0,
    shadowSamples: number = 1,
    reflectionEnabled: number = 0,
    instanceCount: number = 0,
): void {
    const aspect = width / height;
    const proj =
        Camera.mode[eid] === CameraMode.Orthographic
            ? orthographic(Camera.size[eid], aspect, Camera.near[eid], Camera.far[eid], _proj)
            : perspective(Camera.fov[eid], aspect, Camera.near[eid], Camera.far[eid], _proj);
    const world = WorldTransform.data.subarray(eid * 16, eid * 16 + 16);
    const view = invert(world, _view);
    multiply(proj, view, viewProj);

    sceneStaging.set(viewProj, 0);

    const ivp = invertMatrix(viewProj, _ivp);
    sceneStaging.set(ivp, 16);

    sceneStaging.set(world, 32);

    const clearColor = unpackColor(Camera.clearColor[eid]);
    sceneStaging[60] = clearColor.r;
    sceneStaging[61] = clearColor.g;
    sceneStaging[62] = clearColor.b;
    sceneStaging[63] = 1.0;

    sceneStaging[64] = Camera.mode[eid];
    sceneStaging[65] = Camera.size[eid];
    sceneStaging[66] = width;
    sceneStaging[67] = height;

    sceneStaging[68] = Camera.fov[eid];
    sceneStaging[69] = Camera.near[eid];
    sceneStaging[70] = Camera.far[eid];
    sceneStaging[71] = shadowSoftness;
    sceneStagingU32[72] = shadowSamples;
    sceneStagingU32[73] = reflectionEnabled;
    sceneStagingU32[74] = 0;
    sceneStagingU32[75] = instanceCount;
}

const skyArrayBuffer = new ArrayBuffer(SKY_UNIFORM_SIZE);
const skyF32 = new Float32Array(skyArrayBuffer);

export function uploadSky(
    device: GPUDevice,
    buffer: GPUBuffer,
    haze?: HazeParams,
    sky?: SkyParams,
    moon?: MoonParams,
    stars?: StarsParams,
    clouds?: CloudsParams,
    sun?: SunParams,
): void {
    skyF32[0] = haze?.density ?? 0;
    skyF32[1] = sky?.band ?? 0;
    skyF32[2] = 0;
    skyF32[3] = 0;

    const hazeC = unpackColor(haze?.color ?? 0x8090b0);
    skyF32[4] = hazeC.r;
    skyF32[5] = hazeC.g;
    skyF32[6] = hazeC.b;
    skyF32[7] = 1.0;

    const zenithC = unpackColor(sky?.zenith ?? 0);
    skyF32[8] = zenithC.r;
    skyF32[9] = zenithC.g;
    skyF32[10] = zenithC.b;
    skyF32[11] = sky ? 1.0 : 0.0;

    const horizonC = unpackColor(sky?.horizon ?? 0);
    skyF32[12] = horizonC.r;
    skyF32[13] = horizonC.g;
    skyF32[14] = horizonC.b;
    skyF32[15] = 1.0;

    skyF32[16] = moon?.phase ?? 0.5;
    skyF32[17] = moon?.opacity ?? 1.0;
    skyF32[18] = moon ? 1.0 : 0.0;
    skyF32[19] = 0.0;

    const moonAzimuth = ((moon?.azimuth ?? 45) * Math.PI) / 180;
    const moonElevation = ((moon?.elevation ?? 30) * Math.PI) / 180;
    const moonCosEl = Math.cos(moonElevation);
    skyF32[20] = Math.sin(moonAzimuth) * moonCosEl;
    skyF32[21] = Math.sin(moonElevation);
    skyF32[22] = Math.cos(moonAzimuth) * moonCosEl;
    skyF32[23] = 0.0;

    skyF32[24] = stars?.intensity ?? 0.8;
    skyF32[25] = stars?.amount ?? 0.5;
    skyF32[26] = stars ? 1.0 : 0.0;
    skyF32[27] = 0.0;

    skyF32[28] = clouds?.coverage ?? 0;
    skyF32[29] = clouds?.density ?? 0;
    skyF32[30] = clouds?.height ?? 0;
    skyF32[31] = clouds ? 1.0 : 0.0;

    const cloudC = unpackColor(clouds?.color ?? 0xffffff);
    skyF32[32] = cloudC.r;
    skyF32[33] = cloudC.g;
    skyF32[34] = cloudC.b;
    skyF32[35] = 0.0;

    skyF32[36] = sun?.size ?? 0.7;
    skyF32[37] = sun ? 1.0 : 0.0;
    skyF32[38] = sun && sun.color !== 0 ? 1.0 : 0.0;
    skyF32[39] = sun?.glow ?? 0;

    const sunC = unpackColor(sun?.color ?? 0xffffff);
    skyF32[40] = sunC.r;
    skyF32[41] = sunC.g;
    skyF32[42] = sunC.b;
    skyF32[43] = 0.0;

    const sunAz = ((sun?.azimuth ?? 0) * Math.PI) / 180;
    const sunEl = ((sun?.elevation ?? 45) * Math.PI) / 180;
    const sunCosEl = Math.cos(sunEl);
    skyF32[44] = Math.sin(sunAz) * sunCosEl;
    skyF32[45] = Math.sin(sunEl);
    skyF32[46] = Math.cos(sunAz) * sunCosEl;
    skyF32[47] = 0.0;

    device.queue.writeBuffer(buffer, 0, skyArrayBuffer);
}

const _sunScreen = { u: 0, v: 0, visibility: 0 };

export function projectSunToScreen(
    vp: Float32Array,
    dirX: number,
    dirY: number,
    dirZ: number,
): { u: number; v: number; visibility: number } {
    const sx = -dirX * 1000;
    const sy = -dirY * 1000;
    const sz = -dirZ * 1000;
    const cx = vp[0] * sx + vp[4] * sy + vp[8] * sz + vp[12];
    const cy = vp[1] * sx + vp[5] * sy + vp[9] * sz + vp[13];
    const cw = vp[3] * sx + vp[7] * sy + vp[11] * sz + vp[15];
    if (cw <= 0) {
        _sunScreen.u = 0;
        _sunScreen.v = 0;
        _sunScreen.visibility = 0;
        return _sunScreen;
    }
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    const edge = Math.max(Math.abs(ndcX), Math.abs(ndcY));
    const t = Math.max(0, Math.min(1, (edge - 0.6) / 0.6));
    _sunScreen.u = ndcX * 0.5 + 0.5;
    _sunScreen.v = 1 - (ndcY * 0.5 + 0.5);
    _sunScreen.visibility = Math.max(0, 1 - t * t * (3 - 2 * t));
    return _sunScreen;
}

export function projectActiveSun(
    state: State,
    vp: Float32Array,
): { u: number; v: number; visibility: number } {
    let dirX = 0;
    let dirY = -1;
    let dirZ = 0;

    const dirEid = state.only([DirectionalLight]);
    if (dirEid >= 0) {
        dirX = DirectionalLight.directionX[dirEid];
        dirY = DirectionalLight.directionY[dirEid];
        dirZ = DirectionalLight.directionZ[dirEid];
    }

    const sunEid = state.only([Sun]);
    if (sunEid >= 0) {
        const az = (Sun.azimuth[sunEid] * Math.PI) / 180;
        const el = (Sun.elevation[sunEid] * Math.PI) / 180;
        const cosEl = Math.cos(el);
        dirX = -Math.sin(az) * cosEl;
        dirY = -Math.sin(el);
        dirZ = -Math.cos(az) * cosEl;
    }

    return projectSunToScreen(vp, dirX, dirY, dirZ);
}
