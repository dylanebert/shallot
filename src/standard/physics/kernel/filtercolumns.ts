import type { BodyType, FilterBits } from "../common/types";
import type { WorldState } from "../world/world";
import { SHAPE_STRIDE } from "./shapecolumns";

export function bodyType(world: WorldState, id: number): BodyType {
    return world.bodyStore.typeU[id] as BodyType;
}
export function setBodyType(world: WorldState, id: number, type: BodyType): void {
    world.bodyStore.typeU[id] = type;
}
export function shapeBodyId(world: WorldState, id: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + 29] | 0;
}
export function setShapeBodyId(world: WorldState, id: number, body: number): void {
    world.shapeStore.shapeU[id * SHAPE_STRIDE + 29] = body;
}
export function shapeSensorIndex(world: WorldState, id: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + 41] | 0;
}
export function setShapeSensorIndex(world: WorldState, id: number, sensor: number): void {
    world.shapeStore.shapeU[id * SHAPE_STRIDE + 41] = sensor;
}
export function shapeFilterWord(world: WorldState, id: number, lane: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + lane];
}
export function writeShapeFilter(world: WorldState, id: number, filter: FilterBits): void {
    const u = world.shapeStore.shapeU;
    const o = id * SHAPE_STRIDE;
    u[o + 25] = filter.categoryHi;
    u[o + 26] = filter.categoryLo;
    u[o + 27] = filter.maskHi;
    u[o + 28] = filter.maskLo;
    u[o + 31] = filter.groupIndex;
}
