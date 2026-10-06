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
const filterBytes = new ArrayBuffer(16);
const filterData = new DataView(filterBytes);
const filterWords = new Uint32Array(filterBytes);
export function writeShapeFilterValue(world: WorldState, id: number, filter: Filter): void {
    world.shapeStore.refreshViews();
    filterData.setBigUint64(0, filter.categoryBits, true);
    filterData.setBigUint64(8, filter.maskBits, true);
    kernel(world.ecsState).shapeFilterWrite(
        world.worldId,
        id,
        filterWords[1],
        filterWords[0],
        filterWords[3],
        filterWords[2],
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
