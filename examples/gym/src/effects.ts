import {
    Transform,
    Camera,
    Raytracing,
    Shadows,
    Reflections,
    Viewport,
    Tonemap,
    FXAA,
    Vignette,
    Bloom,
    LensFlare,
    GodRays,
    Posterize,
    Dither,
    Sky,
    Sun,
    Stars,
    Moon,
    Haze,
    Clouds,
    DirectionalLight,
    PointLight,
    Part,
    Shape,
} from "@dylanebert/shallot";
import type { State, System } from "@dylanebert/shallot";
import { BenchConfig, BenchmarkState, PipelineMode } from "./config";
import type { BenchmarkState as BenchmarkStateType, SpawnedLight } from "./config";

export const CAMERA_EFFECTS: Record<string, object> = {
    tonemap: Tonemap,
    fxaa: FXAA,
    vignette: Vignette,
    bloom: Bloom,
    lensflare: LensFlare,
    godrays: GodRays,
    posterize: Posterize,
    dither: Dither,
    shadows: Shadows,
    reflections: Reflections,
};

export const ENV_EFFECTS: Record<string, object> = {
    sky: Sky,
    sun: Sun,
    stars: Stars,
    moon: Moon,
    haze: Haze,
    clouds: Clouds,
};

const EFFECTS = { ...CAMERA_EFFECTS, ...ENV_EFFECTS };
export const EFFECT_NAMES = Object.keys(EFFECTS);

const FLAG_NAMES = ["nosun", "pl1", "pl2", "pl3", "pl4"] as const;
type FlagName = (typeof FLAG_NAMES)[number];

const LIGHT_POSITIONS: [number, number, number][] = [
    [40, 20, 0],
    [-30, 15, 35],
    [0, 25, -40],
    [-35, 10, -30],
];

const LIGHT_COLORS = [0xffcc88, 0x88ccff, 0xff8888, 0x88ff88];
const LIGHT_NAMES = ["pl1", "pl2", "pl3", "pl4"] as const;

export const NOSUN = 1 << 0;
const PL_OFFSET = 1;

function isFlag(name: string): name is FlagName {
    return (FLAG_NAMES as readonly string[]).includes(name);
}

export function namesToConfig(names: string[]): { effects: number; flags: number } {
    let effects = 0;
    let flags = 0;
    for (const name of names) {
        const ei = EFFECT_NAMES.indexOf(name);
        if (ei >= 0) {
            effects |= 1 << ei;
            continue;
        }
        const li = (FLAG_NAMES as readonly string[]).indexOf(name);
        if (li >= 0) flags |= 1 << li;
    }
    return { effects, flags };
}

export function parseEffects(str: string | null): string[] {
    if (!str || str === "none") return [];
    if (str === "all") return [...Object.keys(EFFECTS), ...FLAG_NAMES];
    return str
        .split(",")
        .map((s) => s.trim().toLowerCase())
        .filter((s) => s in EFFECTS || isFlag(s));
}

export function reconcilePipeline(state: State, cam: number, prev: { pipeline: number }): boolean {
    const mode = BenchConfig.pipeline[cam];
    if (mode === prev.pipeline) return false;
    prev.pipeline = mode;

    if (state.hasComponent(cam, Raytracing)) state.removeComponent(cam, Raytracing);
    Viewport.width[cam] = 0;
    Viewport.height[cam] = 0;

    if (mode === PipelineMode.Raytracing) {
        state.addComponent(cam, Raytracing);
    }

    return true;
}

export function reconcileEffects(state: State, cam: number, prev: { effects: number }) {
    const bits = BenchConfig.effects[cam];
    if (bits === prev.effects) return;
    prev.effects = bits;

    const camEffects = Object.values(CAMERA_EFFECTS);
    const camKeys = Object.keys(CAMERA_EFFECTS);
    for (let i = 0; i < camKeys.length; i++) {
        const idx = EFFECT_NAMES.indexOf(camKeys[i]);
        const want = (bits & (1 << idx)) !== 0;
        const has = state.hasComponent(cam, camEffects[i]);
        if (want && !has) state.addComponent(cam, camEffects[i]);
        else if (!want && has) state.removeComponent(cam, camEffects[i]);
    }

    const env = state.only([DirectionalLight]);
    if (env < 0) return;
    const envEffects = Object.values(ENV_EFFECTS);
    const envKeys = Object.keys(ENV_EFFECTS);
    for (let i = 0; i < envKeys.length; i++) {
        const idx = EFFECT_NAMES.indexOf(envKeys[i]);
        const want = (bits & (1 << idx)) !== 0;
        const has = state.hasComponent(env, envEffects[i]);
        if (want && !has) state.addComponent(env, envEffects[i]);
        else if (!want && has) state.removeComponent(env, envEffects[i]);
    }
}

function spawnBulb(state: State, pos: [number, number, number], color: number): number {
    const eid = state.addEntity();
    state.addComponent(eid, Part);
    state.addComponent(eid, Transform);
    Transform.posX[eid] = pos[0];
    Transform.posY[eid] = pos[1];
    Transform.posZ[eid] = pos[2];
    Part.shape[eid] = Shape.Sphere;
    Part.color[eid] = color;
    Part.emission[eid] = color;
    Part.emissionIntensity[eid] = 2;
    Part.sizeX[eid] = 0.3;
    Part.sizeY[eid] = 0.3;
    Part.sizeZ[eid] = 0.3;
    Part.shadows[eid] = 0;
    return eid;
}

function spawnLight(
    state: State,
    pos: [number, number, number],
    color: number,
    intensity: number,
    radius: number,
    shadow: boolean,
): SpawnedLight {
    const eid = state.addEntity();
    state.addComponent(eid, PointLight);
    state.addComponent(eid, Transform);
    Transform.posX[eid] = pos[0];
    Transform.posY[eid] = pos[1];
    Transform.posZ[eid] = pos[2];
    PointLight.color[eid] = color;
    PointLight.intensity[eid] = intensity;
    PointLight.radius[eid] = radius;
    PointLight.shadows[eid] = shadow ? 1 : 0;
    const bulb = spawnBulb(state, pos, color);
    return { eid, bulb, basePos: pos };
}

export function reconcileLighting(
    state: State,
    cam: number,
    res: BenchmarkStateType,
    pipelineChanged: boolean,
) {
    const flagBits = BenchConfig.flags[cam];
    const flagsChanged = flagBits !== res.prev.flags;
    if (!flagsChanged && !pipelineChanged) return;
    res.prev.flags = flagBits;

    for (const sl of res.spawnedLights) {
        state.removeEntity(sl.eid);
        state.removeEntity(sl.bulb);
    }
    res.spawnedLights = [];

    if (flagsChanged) {
        const nosun = (flagBits & NOSUN) !== 0;
        const sunEid = state.only([DirectionalLight]);
        if (sunEid >= 0) {
            DirectionalLight.intensity[sunEid] = nosun ? 0 : 1.2;
        }
    }

    for (let i = 0; i < LIGHT_NAMES.length; i++) {
        if (!(flagBits & (1 << (i + PL_OFFSET)))) continue;
        res.spawnedLights.push(
            spawnLight(state, LIGHT_POSITIONS[i], LIGHT_COLORS[i], 3, 120, true),
        );
    }
}

export const ConfigSystem: System = {
    group: "simulation",
    update(state) {
        const cam = state.only([Camera, BenchConfig]);
        if (cam < 0) return;
        const res = state.getResource(BenchmarkState);
        if (!res) return;

        const pipelineChanged = reconcilePipeline(state, cam, res.prev);
        reconcileEffects(state, cam, res.prev);
        reconcileLighting(state, cam, res, pipelineChanged);
    },
};
