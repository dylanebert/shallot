import { kernel } from "../kernel/kernel";
import type { Shape } from "../shapes/shape";
import type { WorldState } from "./world";

export type Visitor = { shapeId: number; generation: number };
export function createSensor(world: WorldState, shapeId: number): void {
    kernel(world.ecsState).sensorCreate(world.worldId, shapeId);
}
export function destroySensor(world: WorldState, sensorShape: Shape): void {
    kernel(world.ecsState).sensorDestroy(world.worldId, sensorShape);
}
