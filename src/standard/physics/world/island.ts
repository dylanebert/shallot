import { ContactField, contactField } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { applyIslandFixes, islandKernel } from "../kernel/islandcolumns";
import type { Joint } from "../solver/joint";
import { wakeSolverSet } from "./solverset";
import type { WorldState } from "./world";

export function createIsland(world: WorldState, set: number): number {
    return islandKernel(world).islandCreate(set);
}
export function destroyIsland(world: WorldState, id: number): void {
    if (world.splitIslandId === id) world.splitIslandId = NULL_INDEX;
    islandKernel(world).islandDestroy(id);
}
export function unlinkContact(world: WorldState, id: number): void {
    islandKernel(world).islandUnlinkContact(id);
}
function wakeEndpoints(world: WorldState, a: number, b: number): void {
    const bodyA = world.bodies[a],
        bodyB = world.bodies[b];
    if (bodyA.setIndex === SetType.Awake && bodyB.setIndex >= SetType.FirstSleeping)
        wakeSolverSet(world, bodyB.setIndex);
    else if (bodyB.setIndex === SetType.Awake && bodyA.setIndex >= SetType.FirstSleeping)
        wakeSolverSet(world, bodyA.setIndex);
}
export function linkContact(world: WorldState, id: number): void {
    const a = contactField(world, id, ContactField.bodyIdA),
        b = contactField(world, id, ContactField.bodyIdA + 3);
    wakeEndpoints(world, a, b);
    islandKernel(world).islandLinkContact(id, world.bodies[a].islandId, world.bodies[b].islandId);
    applyIslandFixes(world);
}
export function linkJoint(world: WorldState, joint: Joint): void {
    const a = joint.edges[0].bodyId,
        b = joint.edges[1].bodyId;
    wakeEndpoints(world, a, b);
    islandKernel(world).islandLinkJoint(
        joint.jointId,
        a,
        b,
        world.bodies[a].islandId,
        world.bodies[b].islandId,
    );
    applyIslandFixes(world);
}
export function unlinkJoint(world: WorldState, joint: Joint): void {
    islandKernel(world).islandUnlinkJoint(joint.jointId, joint.islandId, joint.islandIndex);
    applyIslandFixes(world);
}
export function splitIsland(world: WorldState, baseId: number): void {
    const k = islandKernel(world);
    const count = world.bodies.length;
    const ptr = k.islandSplitIndices(count);
    const indices = new Int32Array(k.memory.buffer, ptr, count);
    for (let i = 0; i < count; ++i) indices[i] = world.bodies[i].islandIndex;
    k.islandSplit(baseId, ptr, count);
    applyIslandFixes(world);
}
