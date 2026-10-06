import { ContactField, contactField, reclassifyBodyContacts } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    islandArrayCount,
    islandArrayGet,
    islandField,
    setIslandField,
} from "../kernel/islandcolumns";
import { kernel } from "../kernel/kernel";
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
    transferJointColumns,
} from "../kernel/solversetcolumns";
import { wakeSetConstraints } from "../solver/graph";
import type { Joint } from "../solver/joint";

import type { WorldState } from "./world";

export type SolverSet = number;
export function destroySolverSet(world: WorldState, set: number): void {
    releaseSolverSet(world, set);
}

function syncMovedShapes(world: WorldState, source: number, moved: number): void {
    if (moved === NULL_INDEX) return;
    const body = moved;
    if (source === SetType.Awake) {
        syncBodyQuery(world, body);
    }
}

export function wakeSolverSet(world: WorldState, set: number): void {
    world.bodyStore.refreshViews();
    const count = setBodyCount(world, set);
    for (let i = 0; i < count; ++i) {
        const body = kernel(world.ecsState).solverSetBodyId(set, i);
        kernel(world.ecsState).bodyWakeRecord(world.worldId, body);
        syncBodyQuery(world, body);
        let key = bodyField(world, body, BodyField.headContactKey);
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
        reclassifyBodyContacts(world, kernel(world.ecsState).solverSetBodyId(set, i));
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
        } else if (kind === 1) {
            const record = world.joints[id];
            record.setIndex = target;
            record.localIndex = index;
        }
        if (kind === 0) syncBodyQuery(world, id);
    }
}

export function transferBody(
    world: WorldState,
    target: SolverSet,
    source: SolverSet,
    body: number,
): void {
    if (target === source) return;
    world.bodyStore.refreshViews();
    const moved = kernel(world.ecsState).bodyTransfer(world.worldId, body, target, true) | 0;
    syncMovedShapes(world, source, moved);
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
        const body = bodyId;
        const moved = kernel(world.ecsState).bodyTransfer(world.worldId, body, sleep, false) | 0;
        syncMovedShapes(world, SetType.Awake, moved);
        syncBodyQuery(world, body);
        let key = bodyField(world, body, BodyField.headContactKey);
        while (key !== NULL_INDEX) {
            const contact = key >> 1,
                edge = key & 1;
            key = contactField(world, contact, ContactField.nextKeyA + 3 * edge);
            if (contactField(world, contact, ContactField.setIndex) === SetType.Disabled) continue;
            if (contactField(world, contact, ContactField.colorIndex) !== NULL_INDEX) continue;
            const other = contactField(world, contact, ContactField.bodyIdA + 3 * (edge ^ 1));
            if (bodyField(world, other, BodyField.setIndex) === SetType.Awake) continue;
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
        reclassifyBodyContacts(world, islandArrayGet(world, id, 0, i));
    if (world.splitIslandId === id) world.splitIslandId = NULL_INDEX;
}
