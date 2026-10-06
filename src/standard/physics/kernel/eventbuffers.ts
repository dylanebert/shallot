import type { EntityId } from "../common/ids";
import type { WorldState } from "../world/world";
import { kernel } from "./kernel";

export const EventKind = {
    SensorBegin: 0,
    SensorEnd: 1,
    ContactBegin: 2,
    ContactEnd: 3,
    ContactHit: 4,
    Joint: 5,
    BodyMove: 6,
    PendingSensorEnd: 7,
} as const;

export function eventCount(world: WorldState, kind: number): number {
    return kernel(world.ecsState).eventCount(world.worldId, kind);
}

/** A fresh public id from a resident event, including its capture-time world and generation. */
export function eventId(world: WorldState, kind: number, index: number, lane: number): EntityId {
    const k = kernel(world.ecsState);
    const packed = k.eventWord(world.worldId, kind, index, lane + 1);
    return {
        index1: k.eventWord(world.worldId, kind, index, lane),
        world0: packed & 0xffff,
        generation:
            lane === 4 && kind >= EventKind.ContactBegin && kind <= EventKind.ContactHit
                ? k.eventWord(world.worldId, kind, index, lane + 2) >>> 0
                : packed >>> 16,
    };
}
