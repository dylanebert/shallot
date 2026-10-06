import type { WorldState } from "../world/world";
import { kernel } from "./kernel";

export function islandKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
export function islandField(world: WorldState, id: number, field: number): number {
    return islandKernel(world).islandField(id, field);
}
export function setIslandField(world: WorldState, id: number, field: number, value: number): void {
    islandKernel(world).islandSetField(id, field, value);
}
export function islandArrayCount(world: WorldState, id: number, kind: number): number {
    return islandKernel(world).islandArrayCount(id, kind);
}
export function islandArrayGet(
    world: WorldState,
    id: number,
    kind: number,
    index: number,
    lane = 0,
): number {
    return islandKernel(world).islandArrayGet(id, kind, index, lane);
}
export function addIslandBody(world: WorldState, id: number, body: number): void {
    islandKernel(world).islandAddBody(id, body);
}
export function removeIslandBody(world: WorldState, id: number, index: number): void {
    islandKernel(world).islandRemoveBody(id, index);
}

export function splitIslandCandidate(world: WorldState): number {
    return islandKernel(world).islandSplitCandidate();
}
export function setSplitIslandCandidate(world: WorldState, id: number): void {
    islandKernel(world).islandSetSplitCandidate(id);
}
