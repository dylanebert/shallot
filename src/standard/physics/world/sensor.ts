import { kernel } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import type { Shape } from "../shapes/shape";
import type { WorldState } from "./world";

export type Visitor = { shapeId: number; generation: number };
const origin = { x: 0, y: 0, z: 0 };
export function createSensor(world: WorldState, shapeId: number): void {
    kernel(world.ecsState).sensorCreate(world.worldId, shapeId);
}
export function recordSensorHit(world: WorldState, sensorId: number, visitorId: number): void {
    kernel(world.ecsState).sensorRecordHit(world.worldId, sensorId, visitorId);
}
export function overlapSensors(world: WorldState): void {
    queryColumns(world).prepare(origin).sensorOverlap(world.worldId);
}
export function destroySensor(world: WorldState, sensorShape: Shape): void {
    kernel(world.ecsState).sensorDestroy(world.worldId, sensorShape);
}
