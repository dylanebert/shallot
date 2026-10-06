import { ContactField, contactField } from "../collision/contact";
import { SetType } from "../common/constants";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { islandKernel } from "../kernel/islandcolumns";

import type { Joint } from "../solver/joint";
import { wakeSolverSet } from "./solverset";
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
function wakeEndpoints(world: WorldState, a: number, b: number): void {
    const bodyA = a,
        bodyB = b;
    if (
        bodyField(world, bodyA, BodyField.setIndex) === SetType.Awake &&
        bodyField(world, bodyB, BodyField.setIndex) >= SetType.FirstSleeping
    )
        wakeSolverSet(world, bodyField(world, bodyB, BodyField.setIndex));
    else if (
        bodyField(world, bodyB, BodyField.setIndex) === SetType.Awake &&
        bodyField(world, bodyA, BodyField.setIndex) >= SetType.FirstSleeping
    )
        wakeSolverSet(world, bodyField(world, bodyA, BodyField.setIndex));
}
export function linkContact(world: WorldState, id: number): void {
    const a = contactField(world, id, ContactField.bodyIdA),
        b = contactField(world, id, ContactField.bodyIdA + 3);
    wakeEndpoints(world, a, b);
    islandKernel(world).islandLinkContact(
        id,
        bodyField(world, a, BodyField.islandId),
        bodyField(world, b, BodyField.islandId),
    );
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
