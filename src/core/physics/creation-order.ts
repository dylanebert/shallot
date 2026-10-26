import type { Component, World } from "../../engine";

/** Internal membership journal: component index, eid, entity reference, presence. */
export const PhysicsCreationOrder = {
    create: () => ({
        components: [] as Component[],
        entries: [] as number[],
        count: 0,
        recording: false,
    }),
};

export function recordPhysicsMembership(
    world: World,
    kind: number,
    eid: number,
    present: boolean,
): void {
    const journal = world.resource(PhysicsCreationOrder);
    if (!journal.recording) return;
    let at = journal.count;
    journal.entries[at++] = kind;
    journal.entries[at++] = eid;
    journal.entries[at++] = world.ref(eid);
    journal.entries[at++] = Number(present);
    journal.count = at;
}
