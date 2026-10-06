import { ContactField, contactField } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import {
    addIslandBody,
    applyIslandFixes,
    islandArrayCount,
    islandArrayGet,
    islandField,
    islandKernel,
    setIslandField,
} from "../kernel/islandcolumns";
import type { Joint } from "../solver/joint";
import { wakeSolverSet } from "./solverset";
import type { WorldState } from "./world";

export function createIsland(world: WorldState, set: number): number {
    return islandKernel(world).islandCreate(set);
}
export function destroyIsland(world: WorldState, id: number): void {
    if (world.splitIslandId === id) world.splitIslandId = NULL_INDEX;
    islandKernel(world).islandDestroy(id);
}
export function unlinkContact(world: WorldState, id: number): void {
    islandKernel(world).islandUnlinkContact(id);
}
function wakeEndpoints(world: WorldState, a: number, b: number): void {
    const bodyA = world.bodies[a],
        bodyB = world.bodies[b];
    if (bodyA.setIndex === SetType.Awake && bodyB.setIndex >= SetType.FirstSleeping)
        wakeSolverSet(world, bodyB.setIndex);
    else if (bodyB.setIndex === SetType.Awake && bodyA.setIndex >= SetType.FirstSleeping)
        wakeSolverSet(world, bodyA.setIndex);
}
export function linkContact(world: WorldState, id: number): void {
    const a = contactField(world, id, ContactField.bodyIdA),
        b = contactField(world, id, ContactField.bodyIdA + 3);
    wakeEndpoints(world, a, b);
    islandKernel(world).islandLinkContact(id, world.bodies[a].islandId, world.bodies[b].islandId);
    applyIslandFixes(world);
}
export function linkJoint(world: WorldState, joint: Joint): void {
    const a = joint.edges[0].bodyId,
        b = joint.edges[1].bodyId;
    wakeEndpoints(world, a, b);
    islandKernel(world).islandLinkJoint(
        joint.jointId,
        a,
        b,
        world.bodies[a].islandId,
        world.bodies[b].islandId,
    );
    applyIslandFixes(world);
}
export function unlinkJoint(world: WorldState, joint: Joint): void {
    islandKernel(world).islandUnlinkJoint(joint.jointId, joint.islandId, joint.islandIndex);
    applyIslandFixes(world);
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
    const baseBodyIds = Array.from({ length: islandArrayCount(world, baseId, 0) }, (_, i) =>
        islandArrayGet(world, baseId, 0, i),
    );
    const links = (kind: number) =>
        Array.from({ length: islandArrayCount(world, baseId, kind) }, (_, i) => ({
            id: islandArrayGet(world, baseId, kind, i),
            bodyIdA: islandArrayGet(world, baseId, kind, i, 1),
            bodyIdB: islandArrayGet(world, baseId, kind, i, 2),
        }));
    const baseContacts = links(1),
        baseJoints = links(2);

    const baseBodyCount = baseBodyIds.length;
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

    // Union over joints, tracking per-component joint counts.
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
        setIslandField(world, baseId, 3, 0);
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
        islandIds[i] = newIsland;
    }

    // Assign bodies to new islands.
    for (let i = 0; i < baseBodyCount; ++i) {
        const bodyId = baseBodyIds[i];
        const root = findParent(parents, i);
        const newIslandId = islandIds[rootMap[root]];
        addIslandBody(world, newIslandId, bodyId);
    }

    // Assign contacts to the island of their bodies (a static body carries no island id).
    for (let i = 0; i < baseContactCount; ++i) {
        const link = baseContacts[i];
        const bodyA = world.bodies[link.bodyIdA];
        const bodyB = world.bodies[link.bodyIdB];
        const targetIslandId = bodyA.islandId !== NULL_INDEX ? bodyA.islandId : bodyB.islandId;
        islandKernel(world).islandAddContact(targetIslandId, link.id, link.bodyIdA, link.bodyIdB);
        applyIslandFixes(world);
    }

    // Assign joints to the island of their bodies.
    for (let i = 0; i < baseJointCount; ++i) {
        const link = baseJoints[i];
        const bodyA = world.bodies[link.bodyIdA];
        const bodyB = world.bodies[link.bodyIdB];
        const targetIslandId = bodyA.islandId !== NULL_INDEX ? bodyA.islandId : bodyB.islandId;
        islandKernel(world).islandAddJoint(targetIslandId, link.id, link.bodyIdA, link.bodyIdB);
        applyIslandFixes(world);
    }

    // Destroy the now-emptied base island.
    destroyIsland(world, baseId);
}
