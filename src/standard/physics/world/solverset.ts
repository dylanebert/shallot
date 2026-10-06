import { ContactField, contactField, setContactField } from "../collision/contact";
import { bodySimSlot, setSimField, setStateField, simField } from "../kernel/bodycolumns";
import {
    createSolverSet,
    releaseSolverSet,
    setArrayCount,
    setArrayGet,
    setArrayPush,
    setArrayRemove,
    setBodyCount,
    setBodyPush,
    setBodyRemove,
} from "../kernel/solversetcolumns";
// Solver sets: the SoA storage that gives bodies/contacts/islands high memory locality. Ported
// from Box3D's solver_set.c (Erin Catto, MIT). Four fixed roles (constants.ts SetType): static,
// disabled, awake, and one set per sleeping island group. A body's sim lives in its set's bodySims
// column; the awake set additionally holds a bodyStates column and the live islands.
//
// Handles destroy/wake/transferBody — the transfers reachable from body create/destroy/setType. trySleepIsland
// (the sleep transition) handles wake completion by transferring touching contacts back to the graph (via graph.ts).
// transferJoint moves a joint's sim between sets (used by setType).

import { ContactFlags, reclassifyBodyContacts, writeBodySimIndex } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { GRAPH_COLOR_COUNT, OVERFLOW_INDEX, SetType } from "../common/constants";
import { residentPush, residentRemove } from "../kernel/bodycolumns";
import { moveJointRecord, releaseJointArray } from "../kernel/jointcolumns";
import { syncBodyQuery } from "../kernel/shapecolumns";
import {
    addJointToGraph,
    clearGraphBodies,
    removeContactFromGraph,
    wakeSetConstraints,
} from "../solver/graph";
import type { Joint } from "../solver/joint";
import { BODY_TRANSIENT_FLAGS, type Body, identityBodyState } from "./body";
import type { WorldState } from "./world";

export type SolverSet = number;
export function destroySolverSet(world: WorldState, setIndex: number): void {
    releaseJointArray(world, GRAPH_COLOR_COUNT + setIndex);
    releaseSolverSet(world, setIndex);
}

// Wake a solver set. Does not merge islands. Handles non-touching contacts parked in the disabled
// set and (via graph.ts) touching contacts / joints held in the constraint graph.
export function wakeSolverSet(world: WorldState, setIndex: number): void {
    const set = setIndex;
    const awakeSet = SetType.Awake;
    const disabledSet = SetType.Disabled;

    const bodies = world.bodies;

    // The woken bodies enter the resident state region (already sized to the total-body high-water, so
    // their slots exist). Refresh the store's views first — a grow elsewhere may have detached them,
    // and the initial writes below go straight through them.
    world.bodyStore.refreshViews();

    const bodyCount = setBodyCount(world, set);
    for (let i = 0; i < bodyCount; ++i) {
        const simSrc = bodySimSlot(set, i);

        const body = bodies[simField(world, simSrc, "bodyId")];
        body.setIndex = SetType.Awake;
        body.localIndex = setBodyCount(world, awakeSet);
        body.sleepTime = 0;

        // The body enters the awake set as resident sim + state views: marshal the sleeping set's plain
        // `simSrc` into the resident columns and append both views (in lockstep by localIndex).
        const state = identityBodyState();
        setStateField(world, state, "flags", body.flags);
        residentPush(world, state, simSrc, body.headShapeId);
        syncBodyQuery(world, body);

        // move non-touching contacts from disabled set to awake set
        let contactKey = body.headContactKey;
        while (contactKey !== NULL_INDEX) {
            const edgeIndex = contactKey & 1;
            const contactId = contactKey >> 1;

            const contact = contactId;
            contactKey = contactField(world, contact, ContactField.nextKeyA + 3 * edgeIndex);

            if (contactField(world, contact, ContactField.setIndex) !== SetType.Disabled) {
                continue;
            }

            const localIndex = contactField(world, contact, ContactField.localIndex);

            setContactField(world, contact, ContactField.setIndex, SetType.Awake);
            setContactField(
                world,
                contact,
                ContactField.localIndex,
                setArrayCount(world, awakeSet, 0),
            );
            setArrayPush(world, awakeSet, 0, contactId);

            const movedLocalIndex = setArrayRemove(world, disabledSet, 0, localIndex);
            if (movedLocalIndex !== NULL_INDEX) {
                const movedContactIndex = setArrayGet(world, disabledSet, 0, localIndex);
                setContactField(world, movedContactIndex, ContactField.localIndex, localIndex);
            }
        }
    }

    // Transfer touching contacts + joints from the sleeping set to the constraint graph.
    wakeSetConstraints(world, set);

    // transfer islands from sleeping set to awake set
    const islandCount = setArrayCount(world, set, 1);
    for (let i = 0; i < islandCount; ++i) {
        const islandSrc = setArrayGet(world, set, 1, i);
        const island = world.islands[islandSrc];
        island.setIndex = SetType.Awake;
        island.localIndex = setArrayCount(world, awakeSet, 1);
        setArrayPush(world, awakeSet, 1, islandSrc);
    }

    // Re-partition the woken bodies' contacts into the incremental collide lists. Runs after
    // wakeSetConstraints so every touching contact already carries setIndex Awake; a contact between two
    // woken bodies converges once both endpoints are visited (reclassify is idempotent).
    for (let i = 0; i < setBodyCount(world, set); ++i) {
        reclassifyBodyContacts(world, bodies[simField(world, bodySimSlot(set, i), "bodyId")]);
    }

    destroySolverSet(world, setIndex);
}

export function transferBody(
    world: WorldState,
    targetSet: SolverSet,
    sourceSet: SolverSet,
    body: Body,
): void {
    if (targetSet === sourceSet) {
        return;
    }

    const sourceIndex = body.localIndex;
    const sourceSim = bodySimSlot(sourceSet, sourceIndex);
    const targetIndex = setBodyCount(world, targetSet);

    // Add to the target set. The awake set holds resident sim + state views (marshal `sourceSim` into
    // the columns); every other set holds a plain deep copy. At most one of source/target is awake, so
    // at most one resident op runs — refresh the store's views first (a prior grow may have detached
    // them). Transient body flags are cleared on the fresh copy either way (b3_bodyTransientFlags).
    if (targetSet === SetType.Awake) {
        world.bodyStore.refreshViews();
        const state = identityBodyState();
        setStateField(world, state, "flags", body.flags);
        residentPush(world, state, sourceSim, body.headShapeId);
        setSimField(
            world,
            bodySimSlot(targetSet, targetIndex),
            "flags",
            simField(world, bodySimSlot(targetSet, targetIndex), "flags") & ~BODY_TRANSIENT_FLAGS,
        );
    } else {
        setBodyPush(world, targetSet, sourceSim);
        const targetSim = bodySimSlot(targetSet, targetIndex);
        setSimField(
            world,
            targetSim,
            "flags",
            simField(world, targetSim, "flags") & ~BODY_TRANSIENT_FLAGS,
        );
    }

    // Remove from the source set: migrate the resident tail record (awake) or swap-remove the plain
    // array, then fix the moved body's localIndex.
    if (sourceSet === SetType.Awake) {
        world.bodyStore.refreshViews();
        const movedBodyId = residentRemove(world, sourceIndex);
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = sourceIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
            syncBodyQuery(world, movedBody);
        }
    } else {
        const movedIndex = setBodyRemove(world, sourceSet, sourceIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedSim = bodySimSlot(sourceSet, sourceIndex);
            world.bodies[simField(world, movedSim, "bodyId")].localIndex = sourceIndex;
        }
    }

    body.setIndex = targetSet;
    body.localIndex = targetIndex;
    syncBodyQuery(world, body);

    // The body's awake-status may have flipped; re-partition its contacts (setType destroys them first,
    // so this is a no-op there, but keeps the invariant under any transferBody caller).
    reclassifyBodyContacts(world, body);
}

/**
 * Move a joint's sim from one solver set to another (b3TransferJoint). The awake set holds joint sims
 * in the constraint graph (including its real overflow fallback), so awake↔sleeping transfers route
 * through graph.ts. The kernel copies the record into the target array and swap-removes the source.
 */
export function transferJoint(
    world: WorldState,
    targetSet: SolverSet,
    sourceSet: SolverSet,
    joint: Joint,
): void {
    if (targetSet === sourceSet) {
        return;
    }

    if (sourceSet === SetType.Awake && joint.colorIndex !== OVERFLOW_INDEX) {
        clearGraphBodies(world, joint.colorIndex, joint.edges[0].bodyId, joint.edges[1].bodyId);
    }
    if (targetSet === SetType.Awake) {
        addJointToGraph(world, joint);
        joint.setIndex = SetType.Awake;
    } else {
        const destination = moveJointRecord(world, joint, GRAPH_COLOR_COUNT + targetSet);
        joint.setIndex = targetSet;
        joint.localIndex = destination;
        joint.colorIndex = NULL_INDEX;
    }
}

// Put a whole island to sleep: move its bodies, touching contacts, and the island itself into a fresh
// sleeping solver set, and park its non-touching contacts in the disabled set (b3TrySleepIsland).
export function trySleepIsland(world: WorldState, islandId: number): void {
    const island = world.islands[islandId];

    // Cannot sleep an island with a pending split and more than one body.
    if (island.constraintRemoveCount > 0 && island.bodies.length > 1) {
        return;
    }

    // Create a new sleeping solver set.
    const sleepSetId = createSolverSet(world);
    const sleepSet = sleepSetId;

    // Grab awake/disabled after creating the sleep set (solverSets may have grown).
    const awakeSet = SetType.Awake;
    const disabledSet = SetType.Disabled;

    // The island's bodies leave the resident awake column below; refresh the store's views first (a
    // grow this step may have detached them) so the swap-remove migrations read/write live bytes.
    world.bodyStore.refreshViews();

    // Move awake bodies to the sleeping set (shuffles the awake set).
    for (let i = 0; i < island.bodies.length; ++i) {
        const bodyId = island.bodies[i];
        const body = world.bodies[bodyId];

        // The body fell asleep this step; flag its move event so the app can sleep the game object too.
        if (body.bodyMoveIndex !== NULL_INDEX) {
            world.bodyStore.markMoveAsleep(body.bodyMoveIndex);
            body.bodyMoveIndex = NULL_INDEX;
        }

        const awakeBodyIndex = body.localIndex;
        const awakeSim = bodySimSlot(awakeSet, awakeBodyIndex);

        // Preserve the sim, discard its state, then compact the awake row.
        const sleepBodyIndex = setBodyCount(world, sleepSet);
        setBodyPush(world, sleepSet, awakeSim);

        const movedBodyId = residentRemove(world, awakeBodyIndex);
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = awakeBodyIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
            syncBodyQuery(world, movedBody);
        }

        body.setIndex = sleepSetId;
        body.localIndex = sleepBodyIndex;
        syncBodyQuery(world, body);

        // Move the body's non-touching contacts to the disabled set.
        let contactKey = body.headContactKey;
        while (contactKey !== NULL_INDEX) {
            const contactId = contactKey >> 1;
            const edgeIndex = contactKey & 1;
            const contact = contactId;
            contactKey = contactField(world, contact, ContactField.nextKeyA + 3 * edgeIndex);

            if (contactField(world, contact, ContactField.setIndex) === SetType.Disabled) {
                // already moved to the disabled set by another body in the island
                continue;
            }
            if (contactField(world, contact, ContactField.colorIndex) !== NULL_INDEX) {
                // touching contact — moved separately below
                continue;
            }

            // If the other body is still awake it will own moving this contact when it sleeps.
            const otherBodyId = contactField(
                world,
                contact,
                ContactField.bodyIdA + 3 * (edgeIndex ^ 1),
            );
            if (world.bodies[otherBodyId].setIndex === SetType.Awake) {
                continue;
            }

            const localIndex = contactField(world, contact, ContactField.localIndex);
            setContactField(world, contact, ContactField.setIndex, SetType.Disabled);
            setContactField(
                world,
                contact,
                ContactField.localIndex,
                setArrayCount(world, disabledSet, 0),
            );
            setArrayPush(world, disabledSet, 0, contact);

            const movedLocalIndex = setArrayRemove(world, awakeSet, 0, localIndex);
            if (movedLocalIndex !== NULL_INDEX) {
                const movedContactIndex = setArrayGet(world, awakeSet, 0, localIndex);
                setContactField(world, movedContactIndex, ContactField.localIndex, localIndex);
            }
        }
    }

    // Move touching contacts from the graph into the sleeping set (shuffles their graph colors).
    for (let i = 0; i < island.contacts.length; ++i) {
        const contactId = island.contacts[i].contactId;
        const contact = contactId;

        const sleepContactIndex = setArrayCount(world, sleepSet, 0);
        setArrayPush(world, sleepSet, 0, contactId);

        // A touching contact lives in its assigned color's scalar `contacts` (mesh/overflow) or
        // `convexContacts` (a convex contact in a real color); removeContactFromGraph handles both,
        // plus clearing the color's bodySet. Under coloring this is no longer always the overflow color.
        const meshContact =
            (contactField(world, contact, ContactField.flags) & ContactFlags.simMeshContact) !== 0;
        removeContactFromGraph(
            world,
            contactField(world, contact, ContactField.bodyIdA + 3 * 0),
            contactField(world, contact, ContactField.bodyIdA + 3 * 1),
            contactField(world, contact, ContactField.colorIndex),
            contactField(world, contact, ContactField.localIndex),
            meshContact,
        );

        setContactField(world, contact, ContactField.setIndex, sleepSetId);
        setContactField(world, contact, ContactField.colorIndex, NULL_INDEX);
        setContactField(world, contact, ContactField.localIndex, sleepContactIndex);
    }

    // Move the island's joints from the graph into the sleeping set (shuffles the overflow color).
    for (let i = 0; i < island.joints.length; ++i) {
        const jointId = island.joints[i].jointId;
        const joint = world.joints[jointId];
        transferJoint(world, sleepSet, awakeSet, joint);
    }

    // Move the island struct itself to the sleeping set.
    {
        const islandIndex = island.localIndex;
        setArrayPush(world, sleepSet, 1, islandId);

        const movedIslandIndex = setArrayRemove(world, awakeSet, 1, islandIndex);
        if (movedIslandIndex !== NULL_INDEX) {
            const movedIslandId = setArrayGet(world, awakeSet, 1, islandIndex);
            world.islands[movedIslandId].localIndex = islandIndex;
        }

        island.setIndex = sleepSetId;
        island.localIndex = 0;
    }

    // Re-partition the slept bodies' contacts: those that moved to the sleep/disabled set drop out of
    // the collide lists, and any that stayed awake (an awake partner keeps them) demote out of recycle.
    for (let i = 0; i < island.bodies.length; ++i) {
        reclassifyBodyContacts(world, world.bodies[island.bodies[i]]);
    }

    if (world.splitIslandId === islandId) {
        world.splitIslandId = NULL_INDEX;
    }
}
