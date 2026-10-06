import {
    ContactField,
    contactField,
    reclassifyBodyContacts,
    writeBodySimIndex,
} from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { bodySimSlot, simBodyId } from "../kernel/bodycolumns";
import {
    islandArrayCount,
    islandArrayGet,
    islandField,
    setIslandField,
} from "../kernel/islandcolumns";
import { syncBodyQuery } from "../kernel/shapecolumns";
import {
    createSolverSet,
    mergeSetColumns,
    moveSetContact,
    moveSetIsland,
    releaseSolverSet,
    setArrayCount,
    setArrayGet,
    setArrayPush,
    setBodyCount,
    sleepSetContact,
    transferBodyColumns,
    transferJointColumns,
    wakeBodyColumns,
} from "../kernel/solversetcolumns";
import { wakeSetConstraints } from "../solver/graph";
import type { Joint } from "../solver/joint";
import type { Body } from "./body";
import type { WorldState } from "./world";

export type SolverSet = number;
export function destroySolverSet(world: WorldState, set: number): void {
    releaseSolverSet(world, set);
}

function fixMovedBody(world: WorldState, source: number, index: number, moved: number): void {
    if (moved === NULL_INDEX) return;
    const body = world.bodies[moved];
    body.localIndex = index;
    if (source === SetType.Awake) {
        writeBodySimIndex(world, body);
        syncBodyQuery(world, body);
    }
}

// body.c and island.c records stay in TypeScript until their stages. The walk follows solver_set.c;
// the kernel owns the row moves and reports the records whose local indices need fixing.
export function wakeSolverSet(world: WorldState, set: number): void {
    world.bodyStore.refreshViews();
    const count = setBodyCount(world, set);
    for (let i = 0; i < count; ++i) {
        const body = world.bodies[simBodyId(world, bodySimSlot(set, i))];
        body.localIndex = wakeBodyColumns(world, set, i, body.flags, body.headShapeId);
        body.setIndex = SetType.Awake;
        body.sleepTime = 0;
        syncBodyQuery(world, body);
        let key = body.headContactKey;
        while (key !== NULL_INDEX) {
            const id = key >> 1,
                edge = key & 1;
            key = contactField(world, id, ContactField.nextKeyA + 3 * edge);
            if (contactField(world, id, ContactField.setIndex) !== SetType.Disabled) continue;
            moveSetContact(
                world,
                SetType.Disabled,
                contactField(world, id, ContactField.localIndex),
                SetType.Awake,
            );
        }
    }
    wakeSetConstraints(world, set);
    const islands = setArrayCount(world, set, 1);
    for (let i = 0; i < islands; ++i) {
        const id = setArrayGet(world, set, 1, i);
        setIslandField(world, id, 1, setArrayPush(world, SetType.Awake, 1, id));
        setIslandField(world, id, 0, SetType.Awake);
    }
    // Contact classification sees the final graph placement of both endpoints.
    for (let i = 0; i < count; ++i)
        reclassifyBodyContacts(world, world.bodies[simBodyId(world, bodySimSlot(set, i))]);
    destroySolverSet(world, set);
}

export function mergeSolverSets(world: WorldState, first: number, second: number): void {
    const moves = mergeSetColumns(world, first, second);
    const target = moves[0];
    for (let i = 3; i < moves.length; i += 3) {
        const kind = moves[i],
            id = moves[i + 1],
            index = moves[i + 2];
        if (kind === 2) {
            setIslandField(world, id, 0, target);
            setIslandField(world, id, 1, index);
        } else {
            const record = kind === 0 ? world.bodies[id] : world.joints[id];
            record.setIndex = target;
            record.localIndex = index;
        }
        if (kind === 0) syncBodyQuery(world, world.bodies[id]);
    }
}

export function transferBody(
    world: WorldState,
    target: SolverSet,
    source: SolverSet,
    body: Body,
): void {
    if (target === source) return;
    world.bodyStore.refreshViews();
    const index = body.localIndex;
    const result = transferBodyColumns(
        world,
        source,
        index,
        target,
        body.flags,
        body.headShapeId,
        true,
    );
    const destination = result[0],
        moved = result[1] | 0;
    fixMovedBody(world, source, index, moved);
    body.setIndex = target;
    body.localIndex = destination;
    syncBodyQuery(world, body);
    reclassifyBodyContacts(world, body);
}

export function transferJoint(
    world: WorldState,
    target: SolverSet,
    source: SolverSet,
    joint: Joint,
): void {
    if (target === source) return;
    const index = joint.localIndex;
    const result = transferJointColumns(
        world,
        source,
        joint.colorIndex,
        index,
        target,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
    );
    const color = result[0] | 0,
        destination = result[1],
        moved = result[2] | 0;
    if (moved !== NULL_INDEX) world.joints[moved].localIndex = index;
    joint.setIndex = target;
    joint.colorIndex = color;
    joint.localIndex = destination;
}

export function trySleepIsland(world: WorldState, id: number): void {
    if (islandField(world, id, 3) > 0 && islandArrayCount(world, id, 0) > 1) return;
    const sleep = createSolverSet(world);
    world.bodyStore.refreshViews();
    for (let i = 0; i < islandArrayCount(world, id, 0); ++i) {
        const bodyId = islandArrayGet(world, id, 0, i);
        const body = world.bodies[bodyId];
        if (body.bodyMoveIndex !== NULL_INDEX) {
            world.bodyStore.markMoveAsleep(body.bodyMoveIndex);
            body.bodyMoveIndex = NULL_INDEX;
        }
        const index = body.localIndex;
        const result = transferBodyColumns(
            world,
            SetType.Awake,
            index,
            sleep,
            body.flags,
            body.headShapeId,
            false,
        );
        const destination = result[0],
            moved = result[1] | 0;
        fixMovedBody(world, SetType.Awake, index, moved);
        body.setIndex = sleep;
        body.localIndex = destination;
        syncBodyQuery(world, body);
        let key = body.headContactKey;
        while (key !== NULL_INDEX) {
            const contact = key >> 1,
                edge = key & 1;
            key = contactField(world, contact, ContactField.nextKeyA + 3 * edge);
            if (contactField(world, contact, ContactField.setIndex) === SetType.Disabled) continue;
            if (contactField(world, contact, ContactField.colorIndex) !== NULL_INDEX) continue;
            const other = contactField(world, contact, ContactField.bodyIdA + 3 * (edge ^ 1));
            if (world.bodies[other].setIndex === SetType.Awake) continue;
            moveSetContact(
                world,
                SetType.Awake,
                contactField(world, contact, ContactField.localIndex),
                SetType.Disabled,
            );
        }
    }
    for (let i = 0; i < islandArrayCount(world, id, 1); ++i)
        sleepSetContact(world, islandArrayGet(world, id, 1, i), sleep);
    for (let i = 0; i < islandArrayCount(world, id, 2); ++i)
        transferJoint(world, sleep, SetType.Awake, world.joints[islandArrayGet(world, id, 2, i)]);
    const index = islandField(world, id, 1);
    const result = moveSetIsland(world, SetType.Awake, index, sleep);
    const destination = result[0],
        moved = result[1] | 0;
    if (moved !== NULL_INDEX) setIslandField(world, moved, 1, index);
    setIslandField(world, id, 0, sleep);
    setIslandField(world, id, 1, destination);
    for (let i = 0; i < islandArrayCount(world, id, 0); ++i)
        reclassifyBodyContacts(world, world.bodies[islandArrayGet(world, id, 0, i)]);
    if (world.splitIslandId === id) world.splitIslandId = NULL_INDEX;
}
