// Solver sets: the SoA storage that gives bodies/contacts/islands high memory locality. Ported
// from Box3D's solver_set.c (Erin Catto, MIT). Four fixed roles (core.ts SetType): static,
// disabled, awake, and one set per sleeping island group. A body's sim lives in its set's bodySims
// column; the awake set additionally holds a bodyStates column and the live islands.
//
// Stage 7 (lifecycle) ports destroy/wake/transferBody — the transfers reachable from body
// create/destroy/setType. Stage 9 (islands+sleeping) adds trySleepIsland (the sleep transition) and
// completes wake by transferring touching contacts back to the graph (via graph.ts). transferJoint
// moves a joint's sim between sets (used by setType). MergeSolverSets stays seamed for the joints stage.

import { NULL_INDEX, swapRemove } from "./array";
import {
    BODY_TRANSIENT_FLAGS,
    type Body,
    type BodySim,
    type BodyState,
    cloneBodySim,
    identityBodyState,
} from "./body";
import { residentPush, residentRemove } from "./bodycolumns";
import { ContactFlags, reclassifyBodyContacts, writeBodySimIndex } from "./contact";
import { SetType } from "./core";
import {
    addJointToGraph,
    removeContactFromGraph,
    removeJointFromGraph,
    wakeSetConstraints,
} from "./graph";
import { allocId, freeId } from "./ids";
import type { IslandSim } from "./island";
import type { Joint, JointSim } from "./joint";
import type { WorldState } from "./world";

/** Contiguous SoA storage for one solver set (b3SolverSet). */
export type SolverSet = {
    bodySims: BodySim[];
    // Only the awake set has body states.
    bodyStates: BodyState[];
    jointSims: JointSim[];
    // Sleeping sets: all contacts. Awake set: non-touching only. Static/disabled: empty.
    contactIndices: number[];
    islandSims: IslandSim[];
    setIndex: number;
};

/** @returns a fresh empty solver set. */
export function emptySolverSet(): SolverSet {
    return {
        bodySims: [],
        bodyStates: [],
        jointSims: [],
        contactIndices: [],
        islandSims: [],
        setIndex: NULL_INDEX,
    };
}

export function destroySolverSet(world: WorldState, setIndex: number): void {
    const set = world.solverSets[setIndex];
    set.bodySims = [];
    set.bodyStates = [];
    set.contactIndices = [];
    set.jointSims = [];
    set.islandSims = [];
    freeId(world.solverSetIdPool, setIndex);
    set.setIndex = NULL_INDEX;
}

// Wake a solver set. Does not merge islands. Handles non-touching contacts parked in the disabled
// set and (via graph.ts) touching contacts / joints held in the constraint graph.
export function wakeSolverSet(world: WorldState, setIndex: number): void {
    const set = world.solverSets[setIndex];
    const awakeSet = world.solverSets[SetType.Awake];
    const disabledSet = world.solverSets[SetType.Disabled];

    const bodies = world.bodies;

    // The woken bodies enter the resident state region (already sized to the total-body high-water, so
    // their slots exist). Refresh the store's views first — a grow elsewhere may have detached them,
    // and the initial writes below go straight through them.
    world.bodyStore.refreshViews();

    const bodyCount = set.bodySims.length;
    for (let i = 0; i < bodyCount; ++i) {
        const simSrc = set.bodySims[i];

        const body = bodies[simSrc.bodyId];
        body.setIndex = SetType.Awake;
        body.localIndex = awakeSet.bodySims.length;
        body.sleepTime = 0;

        // The body enters the awake set as resident sim + state views: marshal the sleeping set's plain
        // `simSrc` into the resident columns and append both views (in lockstep by localIndex).
        const state = identityBodyState();
        state.flags = body.flags;
        residentPush(
            world.bodyStore,
            awakeSet.bodyStates,
            awakeSet.bodySims,
            state,
            simSrc,
            body.headShapeId,
        );

        // move non-touching contacts from disabled set to awake set
        let contactKey = body.headContactKey;
        while (contactKey !== NULL_INDEX) {
            const edgeIndex = contactKey & 1;
            const contactId = contactKey >> 1;

            const contact = world.contacts[contactId];
            contactKey = contact.edges[edgeIndex].nextKey;

            if (contact.setIndex !== SetType.Disabled) {
                continue;
            }

            const localIndex = contact.localIndex;

            contact.setIndex = SetType.Awake;
            contact.localIndex = awakeSet.contactIndices.length;
            awakeSet.contactIndices.push(contactId);

            const movedLocalIndex = swapRemove(disabledSet.contactIndices, localIndex);
            if (movedLocalIndex !== NULL_INDEX) {
                const movedContactIndex = disabledSet.contactIndices[localIndex];
                world.contacts[movedContactIndex].localIndex = localIndex;
            }
        }
    }

    // Transfer touching contacts + joints from the sleeping set to the constraint graph.
    wakeSetConstraints(world, set);

    // transfer islands from sleeping set to awake set
    const islandCount = set.islandSims.length;
    for (let i = 0; i < islandCount; ++i) {
        const islandSrc = set.islandSims[i];
        const island = world.islands[islandSrc.islandId];
        island.setIndex = SetType.Awake;
        island.localIndex = awakeSet.islandSims.length;
        awakeSet.islandSims.push({ islandId: islandSrc.islandId });
    }

    // Re-partition the woken bodies' contacts into the incremental collide lists. Runs after
    // wakeSetConstraints so every touching contact already carries setIndex Awake; a contact between two
    // woken bodies converges once both endpoints are visited (reclassify is idempotent).
    for (let i = 0; i < set.bodySims.length; ++i) {
        reclassifyBodyContacts(world, bodies[set.bodySims[i].bodyId]);
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
    const sourceSim = sourceSet.bodySims[sourceIndex];
    const targetIndex = targetSet.bodySims.length;

    // Add to the target set. The awake set holds resident sim + state views (marshal `sourceSim` into
    // the columns); every other set holds a plain deep copy. At most one of source/target is awake, so
    // at most one resident op runs — refresh the store's views first (a prior grow may have detached
    // them). Transient body flags are cleared on the fresh copy either way (b3_bodyTransientFlags).
    if (targetSet.setIndex === SetType.Awake) {
        world.bodyStore.refreshViews();
        const state = identityBodyState();
        state.flags = body.flags;
        residentPush(
            world.bodyStore,
            targetSet.bodyStates,
            targetSet.bodySims,
            state,
            sourceSim,
            body.headShapeId,
        );
        targetSet.bodySims[targetIndex].flags &= ~BODY_TRANSIENT_FLAGS;
    } else {
        const targetSim = cloneBodySim(sourceSim);
        targetSim.flags &= ~BODY_TRANSIENT_FLAGS;
        targetSet.bodySims.push(targetSim);
    }

    // Remove from the source set: migrate the resident tail record (awake) or swap-remove the plain
    // array, then fix the moved body's localIndex.
    if (sourceSet.setIndex === SetType.Awake) {
        world.bodyStore.refreshViews();
        const movedBodyId = residentRemove(
            world.bodyStore,
            sourceSet.bodyStates,
            sourceSet.bodySims,
            sourceIndex,
        );
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = sourceIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
        }
    } else {
        const movedIndex = swapRemove(sourceSet.bodySims, sourceIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedSim = sourceSet.bodySims[sourceIndex];
            world.bodies[movedSim.bodyId].localIndex = sourceIndex;
        }
    }

    body.setIndex = targetSet.setIndex;
    body.localIndex = targetIndex;

    // The body's awake-status may have flipped; re-partition its contacts (setType destroys them first,
    // so this is a no-op there, but keeps the invariant under any transferBody caller).
    reclassifyBodyContacts(world, body);
}

/**
 * Move a joint's sim from one solver set to another (b3TransferJoint). The awake set holds joint sims
 * in the constraint graph (force-overflow: the single overflow color), so awake↔sleeping transfers
 * route through graph.ts; sleeping↔sleeping moves the sim between plain jointSims columns. The sim
 * object itself is moved (preserving warmstart impulses), never copied.
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

    const localIndex = joint.localIndex;
    const colorIndex = joint.colorIndex;

    // Retrieve the source sim from the graph (awake) or the set's own column (sleeping).
    let sourceSim: JointSim;
    if (sourceSet.setIndex === SetType.Awake) {
        sourceSim = world.constraintGraph.colors[colorIndex].jointSims[localIndex];
    } else {
        sourceSim = sourceSet.jointSims[localIndex];
    }

    // Create the target and re-home the sim.
    if (targetSet.setIndex === SetType.Awake) {
        addJointToGraph(world, sourceSim, joint);
        joint.setIndex = SetType.Awake;
    } else {
        joint.setIndex = targetSet.setIndex;
        joint.localIndex = targetSet.jointSims.length;
        joint.colorIndex = NULL_INDEX;
        targetSet.jointSims.push(sourceSim);
    }

    // Destroy the source slot.
    if (sourceSet.setIndex === SetType.Awake) {
        removeJointFromGraph(
            world,
            joint.edges[0].bodyId,
            joint.edges[1].bodyId,
            colorIndex,
            localIndex,
        );
    } else {
        const movedIndex = swapRemove(sourceSet.jointSims, localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedJoint = world.joints[sourceSet.jointSims[localIndex].jointId];
            movedJoint.localIndex = localIndex;
        }
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
    const sleepSetId = allocId(world.solverSetIdPool);
    if (sleepSetId === world.solverSets.length) {
        world.solverSets.push(emptySolverSet());
    }
    const sleepSet = world.solverSets[sleepSetId];
    sleepSet.bodySims = [];
    sleepSet.bodyStates = [];
    sleepSet.contactIndices = [];
    sleepSet.jointSims = [];
    sleepSet.islandSims = [];
    sleepSet.setIndex = sleepSetId;

    // Grab awake/disabled after creating the sleep set (solverSets may have grown).
    const awakeSet = world.solverSets[SetType.Awake];
    const disabledSet = world.solverSets[SetType.Disabled];

    // The island's bodies leave the resident awake column below; refresh the store's views first (a
    // grow this step may have detached them) so the swap-remove migrations read/write live bytes.
    world.bodyStore.refreshViews();

    // Move awake bodies to the sleeping set (shuffles the awake set).
    for (let i = 0; i < island.bodies.length; ++i) {
        const bodyId = island.bodies[i];
        const body = world.bodies[bodyId];

        // The body fell asleep this step; flag its move event so the app can sleep the game object too.
        if (body.bodyMoveIndex !== NULL_INDEX) {
            world.bodyMoveEvents[body.bodyMoveIndex].fellAsleep = true;
            body.bodyMoveIndex = NULL_INDEX;
        }

        const awakeBodyIndex = body.localIndex;
        const awakeSim = awakeSet.bodySims[awakeBodyIndex];

        // The sleeping set holds a plain deep copy (view→object marshal via cloneBodySim); the awake
        // set's resident sim + state records then compact via one swap-remove migration.
        const sleepBodyIndex = sleepSet.bodySims.length;
        sleepSet.bodySims.push(cloneBodySim(awakeSim));

        const movedBodyId = residentRemove(
            world.bodyStore,
            awakeSet.bodyStates,
            awakeSet.bodySims,
            awakeBodyIndex,
        );
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = awakeBodyIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
        }

        body.setIndex = sleepSetId;
        body.localIndex = sleepBodyIndex;

        // Move the body's non-touching contacts to the disabled set.
        let contactKey = body.headContactKey;
        while (contactKey !== NULL_INDEX) {
            const contactId = contactKey >> 1;
            const edgeIndex = contactKey & 1;
            const contact = world.contacts[contactId];
            contactKey = contact.edges[edgeIndex].nextKey;

            if (contact.setIndex === SetType.Disabled) {
                // already moved to the disabled set by another body in the island
                continue;
            }
            if (contact.colorIndex !== NULL_INDEX) {
                // touching contact — moved separately below
                continue;
            }

            // If the other body is still awake it will own moving this contact when it sleeps.
            const otherBodyId = contact.edges[edgeIndex ^ 1].bodyId;
            if (world.bodies[otherBodyId].setIndex === SetType.Awake) {
                continue;
            }

            const localIndex = contact.localIndex;
            contact.setIndex = SetType.Disabled;
            contact.localIndex = disabledSet.contactIndices.length;
            disabledSet.contactIndices.push(contact.contactId);

            const movedLocalIndex = swapRemove(awakeSet.contactIndices, localIndex);
            if (movedLocalIndex !== NULL_INDEX) {
                const movedContactIndex = awakeSet.contactIndices[localIndex];
                world.contacts[movedContactIndex].localIndex = localIndex;
            }
        }
    }

    // Move touching contacts from the graph into the sleeping set (shuffles their graph colors).
    for (let i = 0; i < island.contacts.length; ++i) {
        const contactId = island.contacts[i].contactId;
        const contact = world.contacts[contactId];

        const sleepContactIndex = sleepSet.contactIndices.length;
        sleepSet.contactIndices.push(contactId);

        // A touching contact lives in its assigned color's scalar `contacts` (mesh/overflow) or
        // `convexContacts` (a convex contact in a real color); removeContactFromGraph handles both,
        // plus clearing the color's bodySet. Under coloring this is no longer always the overflow color.
        const meshContact = (contact.flags & ContactFlags.simMeshContact) !== 0;
        removeContactFromGraph(
            world,
            contact.edges[0].bodyId,
            contact.edges[1].bodyId,
            contact.colorIndex,
            contact.localIndex,
            meshContact,
        );

        contact.setIndex = sleepSetId;
        contact.colorIndex = NULL_INDEX;
        contact.localIndex = sleepContactIndex;
    }

    // Move the island's joints from the graph into the sleeping set (shuffles the overflow color).
    for (let i = 0; i < island.joints.length; ++i) {
        const jointId = island.joints[i].jointId;
        const joint = world.joints[jointId];
        const colorIndex = joint.colorIndex;
        const localIndex = joint.localIndex;
        const jointColor = world.constraintGraph.colors[colorIndex];
        const awakeJointSim = jointColor.jointSims[localIndex];

        const sleepJointIndex = sleepSet.jointSims.length;
        sleepSet.jointSims.push(awakeJointSim);

        const movedLocalIndex = swapRemove(jointColor.jointSims, localIndex);
        if (movedLocalIndex !== NULL_INDEX) {
            const movedJoint = world.joints[jointColor.jointSims[localIndex].jointId];
            movedJoint.localIndex = localIndex;
        }

        joint.setIndex = sleepSetId;
        joint.colorIndex = NULL_INDEX;
        joint.localIndex = sleepJointIndex;
    }

    // Move the island struct itself to the sleeping set.
    {
        const islandIndex = island.localIndex;
        sleepSet.islandSims.push({ islandId });

        const movedIslandIndex = swapRemove(awakeSet.islandSims, islandIndex);
        if (movedIslandIndex !== NULL_INDEX) {
            const movedIslandId = awakeSet.islandSims[islandIndex].islandId;
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
