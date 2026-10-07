import type { WorldState } from "../world/world";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

/** Nongeometry field codes; generation and flags share word 47. */
export const ShapeField = {
    type: 6,
    nextShapeId: 3,
    bodyId: 1,
    sensorIndex: 4,
    proxyKey: 5,
    id: 0,
    prevShapeId: 2,
    density: 7,
    explosionScale: 8,
    localCentroid: 22,
    generation: 47,
    flags: 56,
    aabbMargin: 9,
} as const;
export const ShapeFlags = {
    enableSensorEvents: 1,
    enableContactEvents: 2,
    enableCustomFiltering: 4,
    enableHitEvents: 8,
    enablePreSolveEvents: 16,
    enlargedAABB: 32,
    enableSpeculativeContact: 64,
} as const;
export function shapeField(world: WorldState, id: number, field: number): number {
    const offset = id * SHAPE_STRIDE + field;
    if (field === ShapeField.generation) return world.shapeStore.shapeU[offset] & 0xffff;
    if (field === ShapeField.flags)
        return (world.shapeStore.shapeU[id * SHAPE_STRIDE + 47] >>> 16) & 0xff;
    return world.shapeStore.shapeU[offset] | 0;
}
/** Read a floating-point nongeometry field from its native f32 lane. */
export function shapeScalar(world: WorldState, id: number, field: number): number {
    return world.shapeStore.shapeF[id * SHAPE_STRIDE + field];
}
export function setShapeField(world: WorldState, id: number, field: number, value: number): void {
    const offset = id * SHAPE_STRIDE + field;
    const u = world.shapeStore.shapeU;
    if (field === ShapeField.generation) {
        u[offset] = (u[offset] & 0xffff0000) | (value & 0xffff);
        return;
    }
    if (field === ShapeField.flags) {
        const packed = id * SHAPE_STRIDE + 47;
        u[packed] = (u[packed] & 0xff00ffff) | ((value & 0xff) << 16);
        return;
    }
    if (field === 7 || field === 8 || field === 9 || (field >= 22 && field <= 24))
        world.shapeStore.shapeF[offset] = value;
    else world.shapeStore.shapeU[offset] = value;
}
export function shapeFlag(world: WorldState, id: number, flag: number): boolean {
    return (shapeField(world, id, ShapeField.flags) & flag) !== 0;
}
export function setShapeFlag(world: WorldState, id: number, flag: number, value: boolean): void {
    kernel(world.ecsState).shapeSetFlag(world.worldId, id, flag, value);
}
