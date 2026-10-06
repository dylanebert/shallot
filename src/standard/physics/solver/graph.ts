import { ContactField, contactField } from "../collision/contact";
import { GRAPH_COLOR_COUNT, SetType } from "../common/constants";
import { jointArrayCount, jointArrayKey, jointAt } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { SolverSet } from "../world/solverset";
import type { WorldState } from "../world/world";
import type { Joint } from "./joint";

function graphKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
const contactViews = new WeakMap<WorldState, Uint32Array[]>();
/** Borrow the kernel's contact ids or packed b3ContactSpec words until the graph or memory grows. */
export function graphContacts(world: WorldState, color: number, scalar = false): Uint32Array {
    const k = graphKernel(world);
    const count = k.graphContactCount(color, +scalar) * (scalar ? 2 : 1);
    const ptr = k.graphContactPtr(color, +scalar);
    let views = contactViews.get(world);
    if (views === undefined) {
        views = [];
        contactViews.set(world, views);
    }
    const index = 2 * color + +scalar;
    let view = views[index];
    if (
        view === undefined ||
        view.buffer !== k.memory.buffer ||
        view.byteOffset !== ptr ||
        view.length !== count
    ) {
        view = new Uint32Array(k.memory.buffer, ptr, count);
        views[index] = view;
    }
    return view;
}
export function addContactToGraph(world: WorldState, contact: number): void {
    const a = contactField(world, contact, ContactField.bodyIdA);
    const b = contactField(world, contact, ContactField.bodyIdA + 3);
    graphKernel(world).graphAddContact(
        contact,
        world.bodies[a].localIndex,
        world.bodies[b].localIndex,
    );
}
export function removeContactFromGraph(
    world: WorldState,
    a: number,
    b: number,
    color: number,
    index: number,
    mesh: boolean,
): void {
    graphKernel(world).graphRemoveContact(a, b, color, index, +mesh);
}
export function createJointInGraph(world: WorldState, joint: Joint): void {
    const k = graphKernel(world);
    joint.colorIndex = k.graphCreateJoint(joint.edges[0].bodyId, joint.edges[1].bodyId);
    joint.localIndex = k.jointArrayCount(joint.colorIndex) - 1;
}
export function addJointToGraph(world: WorldState, joint: Joint): void {
    const k = graphKernel(world);
    const index = joint.localIndex;
    const ptr = k.graphAddJoint(
        jointArrayKey(joint),
        index,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
    );
    const result = new Uint32Array(k.memory.buffer, ptr, 3);
    if (result[2] !== 0xffffffff) world.joints[result[2]].localIndex = index;
    joint.colorIndex = result[0];
    joint.localIndex = result[1];
}
export function clearGraphBodies(world: WorldState, color: number, a: number, b: number): void {
    graphKernel(world).graphClearBodies(color, a, b);
}
export function removeJointFromGraph(
    world: WorldState,
    a: number,
    b: number,
    color: number,
    index: number,
): void {
    const moved = graphKernel(world).graphRemoveJoint(a, b, color, index) >>> 0;
    if (moved !== 0xffffffff) world.joints[moved].localIndex = index;
}
export function wakeSetConstraints(world: WorldState, set: SolverSet): void {
    const k = graphKernel(world);
    const key = GRAPH_COLOR_COUNT + set.setIndex;
    const count = jointArrayCount(world, key);
    const contacts = set.contactIndices.length;
    const ptr = k.graphWakeBuffer(contacts, count);
    let input = new Uint32Array(k.memory.buffer, ptr, 3 * (contacts + count));
    for (let i = 0; i < contacts; ++i) {
        const id = set.contactIndices[i];
        input[3 * i] = id;
        input[3 * i + 1] = world.bodies[contactField(world, id, ContactField.bodyIdA)].localIndex;
        input[3 * i + 2] =
            world.bodies[contactField(world, id, ContactField.bodyIdA + 3)].localIndex;
    }
    for (let i = 0; i < count; ++i) {
        const joint = jointAt(world, key, i);
        const o = 3 * (contacts + i);
        input[o] = joint.jointId;
        input[o + 1] = joint.edges[0].bodyId;
        input[o + 2] = joint.edges[1].bodyId;
    }
    k.graphWake(key, contacts, count);
    input = new Uint32Array(k.memory.buffer, ptr, 3 * (contacts + count));
    for (let i = 0; i < count; ++i) {
        const o = 3 * (contacts + i);
        const joint = world.joints[input[o]];
        joint.setIndex = SetType.Awake;
        joint.colorIndex = input[o + 1];
        joint.localIndex = input[o + 2];
    }
}
