import type { BodyType, Filter } from "../common/types";
import type { WorldState } from "../world/world";
import { BodyField, bodyField } from "./bodyrecords";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

export function bodyType(world: WorldState, id: number): BodyType {
    return bodyField(world, id, BodyField.type) as BodyType;
}
export function shapeBodyId(world: WorldState, id: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + 1] | 0;
}
export function shapeSensorIndex(world: WorldState, id: number): number {
    return world.shapeStore.shapeU[id * SHAPE_STRIDE + 4] | 0;
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
