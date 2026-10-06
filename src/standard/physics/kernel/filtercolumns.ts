import type { BodyType, Filter, FilterBits } from "../common/types";
import type { WorldState } from "../world/world";
import { BodyField, bodyField, setBodyField } from "./bodyrecords";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

export function bodyType(world: WorldState, id: number): BodyType {
    return bodyField(world, id, BodyField.type) as BodyType;
}
export function setBodyType(world: WorldState, id: number, type: BodyType): void {
    setBodyField(world, id, BodyField.type, type);
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
    kernel(world.ecsState).shapeAttachSensor(world.worldId, id, sensor);
}
export function shapeFilterWord(world: WorldState, id: number, lane: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + lane];
}
export function writeShapeFilterValue(
    world: WorldState,
    id: number,
    filter: Filter,
    invokeContacts = false,
): void {
    world.shapeStore.refreshViews();
    const k = kernel(world.ecsState);
    if (invokeContacts)
        k.shapeSetFilter64(
            world.worldId,
            id,
            filter.categoryBits,
            filter.maskBits,
            filter.groupIndex,
        );
    else
        k.shapeFilterWrite64(
            world.worldId,
            id,
            filter.categoryBits,
            filter.maskBits,
            filter.groupIndex,
        );
}
export function writeShapeFilter(world: WorldState, id: number, filter: FilterBits): void {
    kernel(world.ecsState).shapeFilterWrite(
        world.worldId,
        id,
        filter.categoryHi,
        filter.categoryLo,
        filter.maskHi,
        filter.maskLo,
        filter.groupIndex,
    );
}
