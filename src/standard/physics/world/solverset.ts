import { ContactField, contactField, reclassifyBodyContacts } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    islandArrayCount,
    islandArrayGet,
    islandField,
    setIslandField,
    setSplitIslandCandidate,
    splitIslandCandidate,
} from "../kernel/islandcolumns";
import { kernel } from "../kernel/kernel";
import { syncBodyQuery } from "../kernel/shapecolumns";
import {
    createSolverSet,
    moveSetContact,
    moveSetIsland,
    releaseSolverSet,
    sleepSetContact,
} from "../kernel/solversetcolumns";
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
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.solverSetWake(set);
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
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.jointTransfer(joint, target);
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
        transferJoint(world, sleep, SetType.Awake, islandArrayGet(world, id, 2, i));
    const index = islandField(world, id, 1);
    const result = moveSetIsland(world, SetType.Awake, index, sleep);
    const destination = result[0],
        moved = result[1] | 0;
    if (moved !== NULL_INDEX) setIslandField(world, moved, 1, index);
    setIslandField(world, id, 0, sleep);
    setIslandField(world, id, 1, destination);
    for (let i = 0; i < islandArrayCount(world, id, 0); ++i)
        reclassifyBodyContacts(world, islandArrayGet(world, id, 0, i));
    if (splitIslandCandidate(world) === id) setSplitIslandCandidate(world, NULL_INDEX);
}
