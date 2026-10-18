import tgpu, { type TgpuBindGroupLayout, type TgpuFn } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { View } from "../../core/rendering";
import type { World } from "../../engine";
import { AlphaModeCode } from "./material-type";

/** The shaded fragment's world-space position, in metres. */
export const FogInput = d.struct({ worldPosition: d.vec3f }).$name("FogInput");
/** RGB is added radiance in standard's exposed colour space; alpha is transmittance to the point. */
export type FogSource = TgpuFn<(input: typeof FogInput) => d.Vec4f>;
const identityFog = tgpu.fn(
    [FogInput],
    d.vec4f,
)(() => {
    "use gpu";
    return d.vec4f(0, 0, 0, 1);
});
export const fogSlot = tgpu.slot(identityFog).$name("fogSlot");

interface Registration {
    name: string;
    source: FogSource;
    layout: TgpuBindGroupLayout;
    bindGroup(world: World, eid: number, view: View): GPUBindGroup;
}
const registryKey = {
    create: () => ({ registration: null as Registration | null, sealed: false }),
};

/**
 * Register one fog source during plugin initialization, before standard warms. Receives the fragment's
 * world position in metres and returns added RGB radiance (already camera-exposed, like shaded colour)
 * and alpha transmittance. Standard applies it only to Blend, Premultiplied, Add and Multiply colour
 * pipelines; opaque, masked and alpha-to-coverage surfaces do not read it. Group 3 is reserved for the
 * supplied layout, which must have .$idx(3). Before each transparent draw, bindGroup is called with its
 * camera entity and view; it must return a valid group for every standard camera, including opted-out
 * cameras. Re-registering the same name and source is a no-op; a different registration is refused.
 */
export function registerFogSource(
    world: World,
    name: string,
    spec: Omit<Registration, "name">,
): void {
    const registry = world.resource(registryKey);
    const prior = registry.registration;
    if (prior) {
        if (prior.name === name && prior.source === spec.source) return;
        throw new Error(`fog source "${prior.name}" is already registered`);
    }
    if (registry.sealed)
        throw new Error(`fog source registration is closed after standard warm: "${name}"`);
    if (!name) throw new Error("fog source name must not be empty");
    if (spec.layout.index !== 3)
        throw new Error("standard fog source layout must be at reserved bind group 3 (.$idx(3))");
    registry.registration = { name, ...spec };
}
export function fogSource(world: World): Registration | null {
    return world.resource(registryKey).registration;
}
export function sealFogSource(world: World): void {
    world.resource(registryKey).sealed = true;
}

export const unfoggedColor = tgpu.fn(
    [d.vec4f, d.vec3f, d.u32],
    d.vec4f,
)((color) => {
    "use gpu";
    return d.vec4f(color);
});
export const foggedColor = tgpu.fn(
    [d.vec4f, d.vec3f, d.u32],
    d.vec4f,
)((color, worldPosition, mode) => {
    "use gpu";
    const fog = fogSlot.$(FogInput({ worldPosition }));
    if (mode === AlphaModeCode.Add) return d.vec4f(std.mul(color.rgb, fog.a), color.a);
    if (mode === AlphaModeCode.Multiply)
        return d.vec4f(std.mix(d.vec3f(1), color.rgb, fog.a), color.a);
    if (mode === AlphaModeCode.Premultiplied)
        return d.vec4f(std.add(std.mul(color.rgb, fog.a), std.mul(fog.rgb, color.a)), color.a);
    return d.vec4f(std.add(std.mul(color.rgb, fog.a), fog.rgb), color.a);
});
