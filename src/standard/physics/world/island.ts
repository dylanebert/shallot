import { islandKernel } from "../kernel/islandcolumns";

import type { Joint } from "../solver/joint";
import type { WorldState } from "./world";

export function createIsland(world: WorldState, set: number): number {
    return islandKernel(world).islandCreate(set);
}
export function destroyIsland(world: WorldState, id: number): void {
    islandKernel(world).islandDestroy(id);
}
export function unlinkContact(world: WorldState, id: number): void {
    islandKernel(world).islandUnlinkContact(id);
}
export function linkContact(world: WorldState, id: number): void {
    islandKernel(world).contactLinkWorld(world.worldId, id);
}
export function linkJoint(world: WorldState, joint: Joint): void {
    islandKernel(world).jointLink(joint);
}
export function unlinkJoint(world: WorldState, joint: Joint): void {
    islandKernel(world).jointUnlink(joint);
}
export function splitIsland(world: WorldState, baseId: number): void {
    const k = islandKernel(world);
    k.islandSplit(baseId);
}
