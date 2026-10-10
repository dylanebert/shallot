import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    DirectionalLight,
    PointLight,
    SpotLight,
    SunDisk,
    VolumetricLight,
} from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import type { World } from "../../engine";
import { unpackColor } from "../../engine";
import { bitcastF32toU32, octDecodeNormal } from "../../engine/utils";

/** Ten directional records fit in the camera-independent uniform; its consumers size the same binding. */
export const MAX_DIRECTIONAL_LIGHTS = 10;

export const DirectionalLightGpu = d.struct({
    /** Normalized light-travel direction (light toward scene). */
    direction: d.vec4f,
    /** Linear RGB light color. */
    color: d.vec4f,
    /** Illuminance in lux, selected-shadow flag, VolumetricLight presence, SunDisk presence. */
    params: d.vec4f,
    /** SunDisk angular diameter, visual intensity, and per-light glow. */
    disk: d.vec4f,
});

/** The standard light uniform. Directionals are packed in a fixed ten-row array. */
export const LightingGpu = d
    .struct({
        directionalLights: d.arrayOf(DirectionalLightGpu, MAX_DIRECTIONAL_LIGHTS),
        directionalCount: d.u32,
    })
    .$name("Lighting");

/** Lighting UBO size rounded to WGSL uniform-struct alignment. */
export const LIGHTING_UNIFORM_SIZE = Math.ceil(d.sizeOf(LightingGpu) / 16) * 16;

export interface Lighting {
    buffer: GPUBuffer;
    staging: Float32Array;
}

interface LightingResources {
    gpu: Lighting;
    directionalEids: Int32Array;
    colorEids: Int32Array;
    colorPacked: Float64Array;
    colors: Float64Array;
    overflowWarned: boolean;
    shadowConflictWarned: boolean;
}

export const lightingKey = { create: createLightingResources };

function createLightingResources(): LightingResources {
    const backing = new ArrayBuffer(LIGHTING_UNIFORM_SIZE);
    return {
        gpu: { buffer: null!, staging: new Float32Array(backing) },
        directionalEids: new Int32Array(MAX_DIRECTIONAL_LIGHTS).fill(-1),
        colorEids: new Int32Array(MAX_DIRECTIONAL_LIGHTS).fill(-1),
        colorPacked: new Float64Array(MAX_DIRECTIONAL_LIGHTS).fill(-1),
        colors: new Float64Array(MAX_DIRECTIONAL_LIGHTS * 3),
        overflowWarned: false,
        shadowConflictWarned: false,
    };
}

function lightingResources(world: World): LightingResources {
    return world.resource(lightingKey);
}

export function initializeLightingState(world: World): void {
    world.resource(lightingKey);
}

export const Lighting: import("../../engine").Resource<Lighting> = {
    create: (world) => world.resource(lightingKey).gpu,
};

const DIR_STRIDE = d.sizeOf(DirectionalLightGpu) / 4;
const DIR_DIRECTION = d.memoryLayoutOf(DirectionalLightGpu, (light) => light.direction).offset / 4;
const DIR_COLOR = d.memoryLayoutOf(DirectionalLightGpu, (light) => light.color).offset / 4;
const DIR_PARAMS = d.memoryLayoutOf(DirectionalLightGpu, (light) => light.params).offset / 4;
const DIR_DISK = d.memoryLayoutOf(DirectionalLightGpu, (light) => light.disk).offset / 4;
const DIR_COUNT = d.memoryLayoutOf(LightingGpu, (lighting) => lighting.directionalCount).offset / 4;
const DIRECTIONAL_TERMS = [DirectionalLight, GlobalTransform];
const _travelDirection = new Float64Array(3);

/** The normalized light-travel direction from an entity's GlobalTransform local -Z. */
export function directionalTravelDirection(world: World, eid: number, out: Float64Array): void {
    const rotation = world.storage(GlobalTransform).rotation;
    const x = rotation.x.get(eid);
    const y = rotation.y.get(eid);
    const z = rotation.z.get(eid);
    const w = rotation.w.get(eid);
    out[0] = -2 * (x * z + w * y);
    out[1] = -2 * (y * z - w * x);
    out[2] = -1 + 2 * (x * x + y * y);
    const length = Math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]) || 1;
    out[0] /= length;
    out[1] /= length;
    out[2] /= length;
}

/** The brightest enabled directional caster; ties are resolved by lower eid. */
export function shadowDirectionalLight(world: World): number {
    const state = lightingResources(world);
    const light = world.storage(DirectionalLight);
    let selected = -1;
    let brightness = -Infinity;
    let enabled = 0;
    for (const eid of world.query(DIRECTIONAL_TERMS)) {
        if (!light.shadowMapsEnabled.get(eid)) continue;
        enabled++;
        const illuminance = light.illuminance.get(eid);
        if (
            illuminance > brightness ||
            (illuminance === brightness && (selected < 0 || eid < selected))
        ) {
            selected = eid;
            brightness = illuminance;
        }
    }
    if (enabled > 1 && !state.shadowConflictWarned) {
        console.warn(
            `shallot: ${enabled} directional lights request shadows; entity ${selected} casts (brightest illuminance, then lower eid)`,
        );
        state.shadowConflictWarned = true;
    } else if (enabled <= 1) state.shadowConflictWarned = false;
    return selected;
}

/** Read up to ten directionals and pack their transform-derived directions and light features. */
export function writeLighting(world: World): void {
    const gpu = world.resource(Lighting);
    if (!world.gpu.device || !gpu.buffer) return;

    const resources = lightingResources(world);
    const staging = gpu.staging;
    staging.fill(0);
    const lights = world.storage(DirectionalLight);
    let count = 0;
    let total = 0;
    for (const eid of world.query(DIRECTIONAL_TERMS)) {
        if (count < MAX_DIRECTIONAL_LIGHTS) resources.directionalEids[count++] = eid;
        total++;
    }
    if (total > MAX_DIRECTIONAL_LIGHTS && !resources.overflowWarned) {
        console.warn(
            `shallot: ${total} directional lights exceed the ${MAX_DIRECTIONAL_LIGHTS} cap; ${total - MAX_DIRECTIONAL_LIGHTS} ignored`,
        );
        resources.overflowWarned = true;
    } else if (total <= MAX_DIRECTIONAL_LIGHTS) resources.overflowWarned = false;

    const shadow = shadowDirectionalLight(world);
    if (shadow >= 0) {
        let at = -1;
        for (let i = 0; i < count; i++) if (resources.directionalEids[i] === shadow) at = i;
        if (at < 0) {
            if (count < MAX_DIRECTIONAL_LIGHTS) count++;
            at = count - 1;
            resources.directionalEids[at] = shadow;
        }
        const first = resources.directionalEids[0];
        resources.directionalEids[0] = shadow;
        resources.directionalEids[at] = first;
    }

    const u32 = new Uint32Array(staging.buffer, staging.byteOffset, staging.length);
    u32[DIR_COUNT] = count;
    for (let i = 0; i < count; i++) {
        const eid = resources.directionalEids[i];
        const base = i * DIR_STRIDE;
        const direction = base + DIR_DIRECTION;
        directionalTravelDirection(world, eid, _travelDirection);
        staging[direction] = _travelDirection[0];
        staging[direction + 1] = _travelDirection[1];
        staging[direction + 2] = _travelDirection[2];
        staging[direction + 3] = 0;

        const color = lights.color.get(eid);
        if (resources.colorEids[i] !== eid || resources.colorPacked[i] !== color) {
            const rgb = unpackColor(color);
            resources.colorEids[i] = eid;
            resources.colorPacked[i] = color;
            resources.colors[i * 3] = rgb.r;
            resources.colors[i * 3 + 1] = rgb.g;
            resources.colors[i * 3 + 2] = rgb.b;
        }
        const colorAt = base + DIR_COLOR;
        staging[colorAt] = resources.colors[i * 3];
        staging[colorAt + 1] = resources.colors[i * 3 + 1];
        staging[colorAt + 2] = resources.colors[i * 3 + 2];
        staging[colorAt + 3] = 0;

        const params = base + DIR_PARAMS;
        staging[params] = lights.illuminance.get(eid);
        staging[params + 1] = Number(eid === shadow);
        staging[params + 2] = Number(world.has(eid, VolumetricLight));
        const hasDisk = world.has(eid, SunDisk);
        staging[params + 3] = Number(hasDisk);
        if (hasDisk) {
            const disk = world.storage(SunDisk);
            const diskAt = base + DIR_DISK;
            staging[diskAt] = disk.angularSize.get(eid);
            staging[diskAt + 1] = disk.intensity.get(eid);
            staging[diskAt + 2] = disk.glow.get(eid);
        }
    }

    world.gpu.device.queue.writeBuffer(gpu.buffer, 0, staging as Float32Array<ArrayBuffer>);
}

/** the point-light list cap. The compacted list the cull pass bins is fixed-size so standard's binding
 * exists for every surface; overflow warns, never silently truncates. */
export const MAX_POINT_LIGHTS = 256;

/**
 * One compacted point or spot light. `posRange` is xyz world position, w = 1/range²; `color` is linear RGB
 * luminous intensity (lumens / 4π), `a` = the source entity id as f32 (the per-entity hook standard matches
 * shadowed casters on); `params` is x = source radius, y = the spot cone axis oct-packed via bitcast, z/w = the Frostbite
 * spot angular scale/offset (a non-spot writes `(radius, 0, 0, 1)` so the angular factor is 1)
 */
export const PointLightGpu = d.struct({
    posRange: d.vec4f,
    color: d.vec4f,
    params: d.vec4f,
});

/**
 * the compacted point-light list: a count header plus the fixed-cap light array, GPU-written by the light
 * compact pass (`cluster.ts`) from the PointLight/SpotLight table + GlobalTransform, and read by
 * standard's clustered loop and the fog march. There is no CPU light list.
 */
export const PointLights = d.struct({
    count: d.vec4u,
    lights: d.arrayOf(PointLightGpu, MAX_POINT_LIGHTS),
});

/**
 * the compact pass's write view of {@link PointLights}: byte-identical, with the count header's first
 * word atomic so the membership scan can reserve slots. WGSL forbids `atomic` in a read-only binding, so
 * the reader and the writer need distinct schemas; `lighting.test.ts` pins their layouts equal.
 * @internal
 */
export const PointLightsRw = d.struct({
    count: d.arrayOf(d.atomic(d.u32), 4),
    lights: d.arrayOf(PointLightGpu, MAX_POINT_LIGHTS),
});

/**
 * the point-light falloff (Bevy `getDistanceAttenuation`): inverse-square with a smooth
 * window (`smooth = saturate(1 − (d²/r²)²)`, attenuation `smooth² / max(d², radiusSq)`),
 * exactly zero at and past the range, and flat at `1/radiusSq` inside the source sphere
 * (Karis representative point: `radiusSq = 0` would spike toward ∞ at the bulb). One
 * function, both sides: standard's clustered loop and the fog march call it on the GPU, the CPU
 * oracles call it directly — there is no WGSL twin to drift from.
 */
export const distanceAttenuation = tgpu.fn(
    [d.f32, d.f32, d.f32],
    d.f32,
)((distSq, invRangeSq, radiusSq) => {
    "use gpu";
    const factor = distSq * invRangeSq;
    const smoothFactor = std.saturate(1 - factor * factor);
    return (smoothFactor * smoothFactor) / std.max(distSq, radiusSq);
});

/**
 * the spot cone's angular attenuation (Frostbite `getAngleAtt`) for one compacted light and the
 * fragment→light direction `L`: `saturate(cd·scale + offset)²`, 1 inside the inner cone and smoothly 0 at
 * the outer. A plain point light carries `(0, 1)` in `params.zw`, so the early-out returns 1 and the
 * multiply is a no-op. `cd` is the cosine between the oct-packed cone axis (`params.y`) and `-L`.
 * {@link spotParams} is the CPU twin that bakes the `(scale, offset)` pair the compact pass stores.
 */
export const spotFactor = tgpu.fn(
    [PointLightGpu, d.vec3f],
    d.f32,
)((light, L) => {
    "use gpu";
    if (light.params.z === 0) return 1;
    const axis = octDecodeNormal(bitcastF32toU32(light.params.y));
    const cd = -std.dot(axis, L);
    const a = std.saturate(cd * light.params.z + light.params.w);
    return a * a;
});

/**
 * the spot cone's angular-attenuation coefficients (Frostbite `getAngleAtt`, Lagarde 2014) from the
 * inner/outer half-angles (degrees). The FS multiplies the light by `saturate(cd·scale + offset)²`, cd =
 * cos(angle between the cone axis and the light→fragment direction): 1 inside the inner cone, smoothly to
 * 0 at the outer. The GPU compact pass bakes these into the light's `params.zw`; the oracle its WGSL twin
 * is pinned to. `inner == outer` is a hard edge (the divide is clamped), never a NaN.
 */
export function spotParams(innerDeg: number, outerDeg: number): { scale: number; offset: number } {
    const cosInner = Math.cos((innerDeg * Math.PI) / 180);
    const cosOuter = Math.cos((outerDeg * Math.PI) / 180);
    const scale = 1 / Math.max(cosInner - cosOuter, 1e-4);
    return { scale, offset: -cosOuter * scale };
}

const POINT_LIGHT_TERMS = [PointLight, GlobalTransform];
const SPOT_LIGHT_TERMS = [SpotLight, GlobalTransform];

/**
 * warn once per episode when more PointLight entities exist than the list cap:
 * the GPU compact pass drops the excess (count beyond {@link MAX_POINT_LIGHTS}
 * never writes an entry), so the overflow is loud, not silent. A count, not a
 * pack: the light data itself flows GPU-side
 */
export function warnLightOverflow(world: World): void {
    const _lighting = world.resource(lightingKey);

    let count = 0;
    for (const eid of world.query(POINT_LIGHT_TERMS)) if (!world.has(eid, SpotLight)) count++;
    for (const _ of world.query(SPOT_LIGHT_TERMS)) count++;
    if (count > MAX_POINT_LIGHTS) {
        const resources = _lighting;
        if (!resources.overflowWarned) {
            resources.overflowWarned = true;
            console.warn(
                `shallot: ${count} point lights exceed the ${MAX_POINT_LIGHTS} cap; ${count - MAX_POINT_LIGHTS} ignored`,
            );
        }
    } else {
        _lighting.overflowWarned = false;
    }
}
