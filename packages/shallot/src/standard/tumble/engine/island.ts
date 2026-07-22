// Persistent islands of connected awake bodies, joints, and touching contacts. Ported from
// Box3D's island.c (Erin Catto, MIT). An island lives inside a solver set; static bodies are never
// in an island. Contacts/joints are stored as links carrying both body ids inline so the split
// pass never touches b3Contact/b3Joint.
//
// Stage 7 (lifecycle) ports create/destroy — a body gets an island on creation — plus the contact
// unlink that contact destroy needs. Stage 9 (islands+sleeping) adds merge (when a contact links two
// islands), contact linking (b3LinkContact), and the union-find split (b3SplitIsland) that runs when a
// touching contact stops. Joints (the join/split joint halves) are stage 10; island.joints is always
// empty here. Validation is compiled out in the fixture build, so b3ValidateIsland is a no-op.

import { NULL_INDEX, swapRemove } from "./array";
import type { Contact } from "./contact";
import { SetType } from "./core";
import { allocId, freeId } from "./ids";
import type { Joint } from "./joint";
import { wakeSolverSet } from "./solverset";
import type { WorldState } from "./world";

/** Cached contact edge stored in an island for split-time union-find (b3ContactLink). */
export type ContactLink = { contactId: number; bodyIdA: number; bodyIdB: number };

/** Cached joint edge stored in an island (b3JointLink). */
export type JointLink = { jointId: number; bodyIdA: number; bodyIdB: number };

/** The movable island stub stored in a solver set's island column (b3IslandSim). */
export type IslandSim = { islandId: number };

/** A persistent island of connected awake bodies (b3Island). */
export type Island = {
    setIndex: number;
    localIndex: number;
    islandId: number;
    constraintRemoveCount: number;
    bodies: number[];
    contacts: ContactLink[];
    joints: JointLink[];
};

/** @returns a fresh empty island slot value. */
function emptyIsland(): Island {
    return {
        setIndex: NULL_INDEX,
        localIndex: NULL_INDEX,
        islandId: NULL_INDEX,
        constraintRemoveCount: 0,
        bodies: [],
        contacts: [],
        joints: [],
    };
}

export function createIsland(world: WorldState, setIndex: number): Island {
    const islandId = allocId(world.islandIdPool);
    if (islandId === world.islands.length) {
        world.islands.push(emptyIsland());
    }

    const set = world.solverSets[setIndex];
    const island = world.islands[islandId];
    island.setIndex = setIndex;
    island.localIndex = set.islandSims.length;
    island.islandId = islandId;
    island.bodies = [];
    island.contacts = [];
    island.joints = [];
    island.constraintRemoveCount = 0;

    set.islandSims.push({ islandId });
    return island;
}

export function destroyIsland(world: WorldState, islandId: number): void {
    if (world.splitIslandId === islandId) {
        world.splitIslandId = NULL_INDEX;
    }

    // assume island is empty
    const island = world.islands[islandId];
    const set = world.solverSets[island.setIndex];
    {
        const localIndex = island.localIndex;
        const lastIndex = set.islandSims.length - 1;
        const moveIslandId = set.islandSims[lastIndex].islandId;
        set.islandSims[localIndex] = set.islandSims[lastIndex];
        world.islands[moveIslandId].localIndex = localIndex;
        set.islandSims.pop();
    }

    island.constraintRemoveCount = 0;
    island.localIndex = NULL_INDEX;
    island.islandId = NULL_INDEX;
    island.setIndex = NULL_INDEX;
    island.bodies = [];
    island.contacts = [];
    island.joints = [];

    freeId(world.islandIdPool, islandId);
}

/** Unlink a contact from its island when it stops touching or is destroyed (b3UnlinkContact). */
export function unlinkContact(world: WorldState, contact: Contact): void {
    const islandId = contact.islandId;
    const island = world.islands[islandId];

    const removeIndex = contact.islandIndex;
    const movedIndex = swapRemove(island.contacts, removeIndex);
    if (movedIndex !== NULL_INDEX) {
        const movedLink = island.contacts[removeIndex];
        world.contacts[movedLink.contactId].islandIndex = removeIndex;
    }

    contact.islandId = NULL_INDEX;
    contact.islandIndex = NULL_INDEX;
    island.constraintRemoveCount += 1;
}

// Merge two islands, keeping the larger to reduce reshuffling; the smaller is emptied and destroyed
// (b3MergeIslands). @returns the surviving island id. Handles null ids (a static-body edge).
function mergeIslands(world: WorldState, islandIdA: number, islandIdB: number): number {
    if (islandIdA === islandIdB) {
        return islandIdA;
    }
    if (islandIdA === NULL_INDEX) {
        return islandIdB;
    }
    if (islandIdB === NULL_INDEX) {
        return islandIdA;
    }

    let bigIsland: Island;
    let smallIsland: Island;
    {
        const islandA = world.islands[islandIdA];
        const islandB = world.islands[islandIdB];
        // Keep the biggest island to reduce cache misses
        if (islandA.bodies.length >= islandB.bodies.length) {
            bigIsland = islandA;
            smallIsland = islandB;
        } else {
            bigIsland = islandB;
            smallIsland = islandA;
        }
    }

    const bigIslandId = bigIsland.islandId;

    // Move bodies from smaller island to larger island
    for (let i = 0; i < smallIsland.bodies.length; ++i) {
        const bodyId = smallIsland.bodies[i];
        const body = world.bodies[bodyId];
        body.islandId = bigIslandId;
        body.islandIndex = bigIsland.bodies.length;
        bigIsland.bodies.push(bodyId);
    }

    // Migrate contacts from smaller island to larger island
    for (let i = 0; i < smallIsland.contacts.length; ++i) {
        const link = smallIsland.contacts[i];
        const contact = world.contacts[link.contactId];
        contact.islandId = bigIslandId;
        contact.islandIndex = bigIsland.contacts.length;
        bigIsland.contacts.push(link);
    }

    // Migrate joints from smaller island to larger island (stage 10; always empty here)
    for (let i = 0; i < smallIsland.joints.length; ++i) {
        const link = smallIsland.joints[i];
        const joint = world.joints[link.jointId];
        joint.islandId = bigIslandId;
        joint.islandIndex = bigIsland.joints.length;
        bigIsland.joints.push(link);
    }

    bigIsland.constraintRemoveCount += smallIsland.constraintRemoveCount;

    destroyIsland(world, smallIsland.islandId);

    return bigIslandId;
}

// Add a touching contact to an island's contact link list (b3AddContactToIsland).
function addContactToIsland(world: WorldState, islandId: number, contact: Contact): void {
    const island = world.islands[islandId];
    contact.islandId = islandId;
    contact.islandIndex = island.contacts.length;
    island.contacts.push({
        contactId: contact.contactId,
        bodyIdA: contact.edges[0].bodyId,
        bodyIdB: contact.edges[1].bodyId,
    });
}

// Link a touching contact into an island, merging the two bodies' islands (b3LinkContact). Wakes a
// sleeping body whose partner is awake so the merged island lives in the awake set.
export function linkContact(world: WorldState, contact: Contact): void {
    const bodyIdA = contact.edges[0].bodyId;
    const bodyIdB = contact.edges[1].bodyId;
    const bodyA = world.bodies[bodyIdA];
    const bodyB = world.bodies[bodyIdB];

    // Wake the sleeping body if the other is awake.
    if (bodyA.setIndex === SetType.Awake && bodyB.setIndex >= SetType.FirstSleeping) {
        wakeSolverSet(world, bodyB.setIndex);
    }
    if (bodyB.setIndex === SetType.Awake && bodyA.setIndex >= SetType.FirstSleeping) {
        wakeSolverSet(world, bodyA.setIndex);
    }

    const islandIdA = bodyA.islandId;
    const islandIdB = bodyB.islandId;

    // Merge islands. This destroys one of the islands.
    const finalIslandId = mergeIslands(world, islandIdA, islandIdB);

    // Add contact to the island that survived
    addContactToIsland(world, finalIslandId, contact);
}

// Add a joint to an island's joint link list (b3AddJointToIsland).
function addJointToIsland(world: WorldState, islandId: number, joint: Joint): void {
    const island = world.islands[islandId];
    joint.islandId = islandId;
    joint.islandIndex = island.joints.length;
    island.joints.push({
        jointId: joint.jointId,
        bodyIdA: joint.edges[0].bodyId,
        bodyIdB: joint.edges[1].bodyId,
    });
}

// Link a joint into an island, merging the two bodies' islands (b3LinkJoint). Wakes a sleeping body
// whose partner is awake so the merged island lives in the awake set.
export function linkJoint(world: WorldState, joint: Joint): void {
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];

    if (bodyA.setIndex === SetType.Awake && bodyB.setIndex >= SetType.FirstSleeping) {
        wakeSolverSet(world, bodyB.setIndex);
    } else if (bodyB.setIndex === SetType.Awake && bodyA.setIndex >= SetType.FirstSleeping) {
        wakeSolverSet(world, bodyA.setIndex);
    }

    const islandIdA = bodyA.islandId;
    const islandIdB = bodyB.islandId;

    // Merge islands. This destroys one of the islands.
    const finalIslandId = mergeIslands(world, islandIdA, islandIdB);

    // Add joint to the island that survived
    addJointToIsland(world, finalIslandId, joint);
}

/** Unlink a joint from its island when it is destroyed (b3UnlinkJoint). */
export function unlinkJoint(world: WorldState, joint: Joint): void {
    if (joint.islandId === NULL_INDEX) {
        return;
    }

    const islandId = joint.islandId;
    const island = world.islands[islandId];

    const removeIndex = joint.islandIndex;
    const movedIndex = swapRemove(island.joints, removeIndex);
    if (movedIndex !== NULL_INDEX) {
        const movedLink = island.joints[removeIndex];
        world.joints[movedLink.jointId].islandIndex = removeIndex;
    }

    joint.islandId = NULL_INDEX;
    joint.islandIndex = NULL_INDEX;
    island.constraintRemoveCount += 1;
}

// --- Union-find island split -----------------------------------------------------------------

// Find the root of a node's component, halving the path for later queries (b3IslandFindParent).
function findParent(parents: number[], node: number): number {
    while (parents[node] !== node) {
        const grandParent = parents[parents[node]];
        parents[node] = grandParent;
        node = grandParent;
    }
    return node;
}

// Union the components of node1 and node2, tracking per-component contact/joint counts (b3IslandUnion).
function islandUnion(
    parents: number[],
    ranks: number[],
    node1: number,
    node2: number,
    contactCounts: number[],
    jointCounts: number[],
): void {
    const root1 = findParent(parents, node1);
    const root2 = findParent(parents, node2);
    if (root1 === root2) {
        return;
    }
    if (ranks[root1] < ranks[root2]) {
        parents[root1] = root2;
        contactCounts[root2] += contactCounts[root1];
        jointCounts[root2] += jointCounts[root1];
    } else if (ranks[root1] > ranks[root2]) {
        parents[root2] = root1;
        contactCounts[root1] += contactCounts[root2];
        jointCounts[root1] += jointCounts[root2];
    } else {
        parents[root2] = root1;
        ranks[root1] += 1;
        contactCounts[root1] += contactCounts[root2];
        jointCounts[root1] += jointCounts[root2];
    }
}

// Split an island into its connected components after some contacts/joints were removed
// (b3SplitIsland). Uses union-find over the surviving contact/joint links; static bodies (null island
// index) don't connect components. A no-op that only clears constraintRemoveCount when still connected.
export function splitIsland(world: WorldState, baseId: number): void {
    const baseIsland = world.islands[baseId];

    const baseBodyCount = baseIsland.bodies.length;
    const baseBodyIds = baseIsland.bodies;
    const baseContacts = baseIsland.contacts;
    const baseJoints = baseIsland.joints;
    const baseContactCount = baseContacts.length;
    const baseJointCount = baseJoints.length;

    const parents: number[] = new Array(baseBodyCount);
    const ranks: number[] = new Array(baseBodyCount);
    const contactCounts: number[] = new Array(baseBodyCount);
    const jointCounts: number[] = new Array(baseBodyCount);
    for (let i = 0; i < baseBodyCount; ++i) {
        parents[i] = i;
        ranks[i] = 0;
        contactCounts[i] = 0;
        jointCounts[i] = 0;
    }

    const bodies = world.bodies;

    // Union over contacts, tracking per-component contact counts.
    for (let i = 0; i < baseContactCount; ++i) {
        const bodyA = bodies[baseContacts[i].bodyIdA];
        const bodyB = bodies[baseContacts[i].bodyIdB];
        const islandIndexA = bodyA.islandIndex;
        const islandIndexB = bodyB.islandIndex;

        if (islandIndexA !== NULL_INDEX && islandIndexB !== NULL_INDEX) {
            islandUnion(parents, ranks, islandIndexA, islandIndexB, contactCounts, jointCounts);
            const root = findParent(parents, islandIndexA);
            contactCounts[root] += 1;
        } else {
            const islandIndex = islandIndexA !== NULL_INDEX ? islandIndexA : islandIndexB;
            const root = findParent(parents, islandIndex);
            contactCounts[root] += 1;
        }
    }

    // Union over joints, tracking per-component joint counts (stage 10; always empty here).
    for (let i = 0; i < baseJointCount; ++i) {
        const bodyA = bodies[baseJoints[i].bodyIdA];
        const bodyB = bodies[baseJoints[i].bodyIdB];
        const islandIndexA = bodyA.islandIndex;
        const islandIndexB = bodyB.islandIndex;

        if (islandIndexA !== NULL_INDEX && islandIndexB !== NULL_INDEX) {
            islandUnion(parents, ranks, islandIndexA, islandIndexB, contactCounts, jointCounts);
            const root = findParent(parents, islandIndexA);
            jointCounts[root] += 1;
        } else {
            const islandIndex = islandIndexA !== NULL_INDEX ? islandIndexA : islandIndexB;
            const root = findParent(parents, islandIndex);
            jointCounts[root] += 1;
        }
    }

    // Flatten all parent indices and count connected components.
    let componentCount = 0;
    for (let i = 0; i < baseBodyCount; ++i) {
        parents[i] = findParent(parents, i);
        if (parents[i] === i) {
            componentCount += 1;
        }
    }

    // Early return — island is still fully connected, no split needed.
    if (componentCount === 1) {
        baseIsland.constraintRemoveCount = 0;
        return;
    }

    // Map from body index to new island index (only set for root bodies).
    const rootMap: number[] = new Array(baseBodyCount).fill(NULL_INDEX);
    let islandCount = 0;
    for (let i = 0; i < baseBodyCount; ++i) {
        const rootIndex = parents[i];
        if (rootMap[rootIndex] === NULL_INDEX) {
            rootMap[rootIndex] = islandCount;
            islandCount += 1;
        }
    }

    // Create the new islands (this pushes islandSims; baseIsland's own local index is unaffected).
    const islandIds: number[] = new Array(islandCount);
    for (let i = 0; i < islandCount; ++i) {
        const newIsland = createIsland(world, SetType.Awake);
        islandIds[i] = newIsland.islandId;
    }

    // Assign bodies to new islands.
    for (let i = 0; i < baseBodyCount; ++i) {
        const bodyId = baseBodyIds[i];
        const root = findParent(parents, i);
        const newIslandId = islandIds[rootMap[root]];
        const body = world.bodies[bodyId];
        const newIsland = world.islands[newIslandId];
        body.islandId = newIslandId;
        body.islandIndex = newIsland.bodies.length;
        newIsland.bodies.push(bodyId);
    }

    // Assign contacts to the island of their bodies (a static body carries no island id).
    for (let i = 0; i < baseContactCount; ++i) {
        const link = baseContacts[i];
        const contact = world.contacts[link.contactId];
        const bodyA = world.bodies[link.bodyIdA];
        const bodyB = world.bodies[link.bodyIdB];
        const targetIslandId = bodyA.islandId !== NULL_INDEX ? bodyA.islandId : bodyB.islandId;
        const targetIsland = world.islands[targetIslandId];
        contact.islandId = targetIslandId;
        contact.islandIndex = targetIsland.contacts.length;
        targetIsland.contacts.push(link);
    }

    // Assign joints to the island of their bodies (stage 10; always empty here).
    for (let i = 0; i < baseJointCount; ++i) {
        const link = baseJoints[i];
        const joint = world.joints[link.jointId];
        const bodyA = world.bodies[link.bodyIdA];
        const bodyB = world.bodies[link.bodyIdB];
        const targetIslandId = bodyA.islandId !== NULL_INDEX ? bodyA.islandId : bodyB.islandId;
        const targetIsland = world.islands[targetIslandId];
        joint.islandId = targetIslandId;
        joint.islandIndex = targetIsland.joints.length;
        targetIsland.joints.push(link);
    }

    // Destroy the now-emptied base island.
    destroyIsland(world, baseId);
}
