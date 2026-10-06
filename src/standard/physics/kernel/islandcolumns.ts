import type { WorldState } from "../world/world";
import { type Kernel, kernel } from "./kernel";

const fixViews = new WeakMap<Kernel, Int32Array>();
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
export function applyIslandFixes(world: WorldState): void {
    const k = islandKernel(world);
    const count = k.islandFixCount();
    if (count === 0) return;
    const buffer = k.memory.buffer;
    let fixes = fixViews.get(k);
    if (fixes?.buffer !== buffer) {
        fixes = new Int32Array(buffer);
        fixViews.set(k, fixes);
    }
    // Borrow the memory-wide view: the fix vector can relocate without memory growing.
    const start = k.islandFixData() >>> 2;
    for (let i = start; i < start + count; i += 4) {
        const record = fixes[i] === 0 ? world.bodies[fixes[i + 1]] : world.joints[fixes[i + 1]];
        record.islandId = fixes[i + 2];
        record.islandIndex = fixes[i + 3];
        if (world.splitIslandId !== -1 && islandField(world, world.splitIslandId, 2) === -1)
            world.splitIslandId = -1;
    }
    k.islandFixClear();
}
export function addIslandBody(world: WorldState, id: number, body: number): void {
    islandKernel(world).islandAddBody(id, body);
    applyIslandFixes(world);
}
export function removeIslandBody(world: WorldState, id: number, index: number): void {
    islandKernel(world).islandRemoveBody(id, index);
    applyIslandFixes(world);
}
