import type { EntityId } from "../common/ids";
import { kernel } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import type { Shape } from "../shapes/shape";
import type { WorldState } from "./world";

export type Visitor = { shapeId: number; generation: number };
export type SensorBeginTouchEvent = { sensorShapeId: EntityId; visitorShapeId: EntityId };
const origin = { x: 0, y: 0, z: 0 };
export function createSensor(world: WorldState, shapeId: number): void {
    kernel(world.ecsState).sensorCreate(world.worldId, shapeId);
}
export function recordSensorHit(world: WorldState, sensorId: number, visitorId: number): void {
    kernel(world.ecsState).sensorRecordHit(world.worldId, sensorId, visitorId);
}
function deliver(world: WorldState): void {
    const k = kernel(world.ecsState);
    for (let i = 0, count = k.sensorEventCount(world.worldId); i < count; ++i) {
        const event = {
            sensorShapeId: {
                index1: k.sensorEventWord(world.worldId, i, 1) + 1,
                world0: world.worldId,
                generation: k.sensorEventWord(world.worldId, i, 2),
            },
            visitorShapeId: {
                index1: k.sensorEventWord(world.worldId, i, 3) + 1,
                world0: world.worldId,
                generation: k.sensorEventWord(world.worldId, i, 4),
            },
        };
        if (k.sensorEventWord(world.worldId, i, 0))
            world.sensorEndEvents[world.endEventArrayIndex].push(event);
        else world.sensorBeginEvents.push(event);
    }
}
export function overlapSensors(world: WorldState): void {
    const k = queryColumns(world).prepare(origin);
    k.sensorOverlap(world.worldId);
    deliver(world);
}
export function destroySensor(world: WorldState, sensorShape: Shape): void {
    kernel(world.ecsState).sensorDestroy(world.worldId, sensorShape);
    deliver(world);
}
