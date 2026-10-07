import { ContactField, contactField } from "../collision/contact";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";

function graphKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
const contactViews = new WeakMap<WorldState, Uint32Array[]>();
const _wakeViews = new WeakMap<WorldState, Uint32Array>();
/** Borrow the kernel's contact ids or packed b3ContactSpec words until the graph or memory grows. */
export function graphContacts(world: WorldState, color: number, scalar = false): Uint32Array {
    const k = graphKernel(world);
    const count = k.graphContactCount(color, +scalar) * (scalar ? 3 : 1);
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
        bodyField(world, a, BodyField.localIndex),
        bodyField(world, b, BodyField.localIndex),
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
export function clearGraphBodies(world: WorldState, color: number, a: number, b: number): void {
    graphKernel(world).graphClearBodies(color, a, b);
}
