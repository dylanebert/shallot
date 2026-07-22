// Constraint graph — Box3D's constraint_graph.c (Erin Catto, MIT). Awake *touching* contacts (and
// joints) are distributed across solver colors by greedy graph coloring, so a color's constraints
// share no dynamic body and can be solved in parallel lanes (the wide solver). Dynamic-dynamic
// constraints take colors 0..DYNAMIC_COLOR_COUNT-1; dynamic-static constraints build from the high
// end (OVERFLOW_INDEX-1 down to 1) for higher solver priority; anything that fits no color spills to
// the single serial overflow color. Mesh/height-field contacts and the overflow color are always
// solved scalar (the `contacts` list); a convex contact in a real color is solved wide
// (`convexContacts`). A color's `bodySet` tracks which bodies it already constrains (unused on the
// overflow color, which imposes no sharing limit).
//
// FORCE_OVERFLOW mirrors C's `#if B3_FORCE_OVERFLOW` parity knob: when set, every constraint routes
// to the overflow color and the coloring bit sets go unused — the shape the port shipped against the
// force-overflow fixtures. It flips to false with the default-config fixture migration (the colored,
// wide solve). The transient per-step constraint arrays live in wasm columns, not here.
//
// Coloring is integer-only, so no fround discipline applies here.

import { NULL_INDEX, swapRemove } from "./array";
import {
    type BitSet,
    clearBit,
    createBitSet,
    getBit,
    setBitCountAndClear,
    setBitGrow,
} from "./bitset";
import { type Contact, ContactFlags } from "./contact";
import { DYNAMIC_COLOR_COUNT, GRAPH_COLOR_COUNT, OVERFLOW_INDEX, SetType } from "./core";
import { emptyJointSim, type Joint, type JointSim } from "./joint";
import type { SolverSet } from "./solverset";
import { BodyType } from "./types";
import type { WorldState } from "./world";

// Route every constraint to the serial overflow color, mirroring C's `#if B3_FORCE_OVERFLOW`. The
// default is the colored wide solve (graph coloring + the 4-lane convex path); set true only to fall
// back to the serial overflow path (the parity knob C keeps behind `#if B3_FORCE_OVERFLOW`).
const FORCE_OVERFLOW = false;

/** One touching contact's entry in a graph color (b3ContactSpec). */
export type ContactSpec = { contactId: number; manifoldStart: number; manifoldCount: number };

/** One solver color: the constraints solved together (b3GraphColor). `bodySet` is unused on the
 * overflow color. */
export type GraphColor = {
    bodySet: BitSet;
    contacts: ContactSpec[];
    convexContacts: number[];
    jointSims: JointSim[];
};

/** The solver constraint graph (b3ConstraintGraph). */
export type ConstraintGraph = { colors: GraphColor[] };

/** @returns a fresh graph with all colors empty (b3CreateGraph). Each non-overflow color's bodySet
 * is sized to the body capacity; the overflow color needs none. */
export function createGraph(bodyCapacity: number): ConstraintGraph {
    const cap = bodyCapacity > 8 ? bodyCapacity : 8;
    const colors: GraphColor[] = [];
    for (let i = 0; i < GRAPH_COLOR_COUNT; ++i) {
        const bodySet = createBitSet(i < OVERFLOW_INDEX ? cap : 0);
        if (i < OVERFLOW_INDEX) {
            setBitCountAndClear(bodySet, cap);
        }
        colors.push({ bodySet, contacts: [], convexContacts: [], jointSims: [] });
    }
    return { colors };
}

/** Flag-gated color assignment (the shared body of b3AddContactToGraph / b3AssignJointColor):
 * overflow under FORCE_OVERFLOW, else the greedy color. */
function assignColor(
    graph: ConstraintGraph,
    bodyIdA: number,
    bodyIdB: number,
    typeA: BodyType,
    typeB: BodyType,
): number {
    return FORCE_OVERFLOW ? OVERFLOW_INDEX : greedyColor(graph, bodyIdA, bodyIdB, typeA, typeB);
}

/** Greedy color for a dynamic-involving constraint. Sets the chosen color's body bits and @returns
 * the color, or the overflow color when none fits. Exported for the coloring unit tests (the live
 * path reaches it through the flag-gated {@link assignColor}). */
export function greedyColor(
    graph: ConstraintGraph,
    bodyIdA: number,
    bodyIdB: number,
    typeA: BodyType,
    typeB: BodyType,
): number {
    if (typeA === BodyType.Dynamic && typeB === BodyType.Dynamic) {
        // Dynamic constraint colors cannot encroach on colors reserved for static constraints.
        for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i) {
            const color = graph.colors[i];
            if (getBit(color.bodySet, bodyIdA) || getBit(color.bodySet, bodyIdB)) {
                continue;
            }
            setBitGrow(color.bodySet, bodyIdA);
            setBitGrow(color.bodySet, bodyIdB);
            return i;
        }
    } else if (typeA === BodyType.Dynamic) {
        // Static constraint colors build from the end for higher priority than dyn-dyn constraints.
        for (let i = OVERFLOW_INDEX - 1; i >= 1; --i) {
            const color = graph.colors[i];
            if (getBit(color.bodySet, bodyIdA)) {
                continue;
            }
            setBitGrow(color.bodySet, bodyIdA);
            return i;
        }
    } else if (typeB === BodyType.Dynamic) {
        for (let i = OVERFLOW_INDEX - 1; i >= 1; --i) {
            const color = graph.colors[i];
            if (getBit(color.bodySet, bodyIdB)) {
                continue;
            }
            setBitGrow(color.bodySet, bodyIdB);
            return i;
        }
    }

    return OVERFLOW_INDEX;
}

/** Clone a touching contact into the constraint graph (b3AddContactToGraph). A convex contact in a
 * real color joins `convexContacts` (wide-solved); a mesh contact or any overflow contact joins
 * `contacts` (scalar). */
export function addContactToGraph(world: WorldState, contact: Contact): void {
    const graph = world.constraintGraph;

    const bodyIdA = contact.edges[0].bodyId;
    const bodyIdB = contact.edges[1].bodyId;
    const bodyA = world.bodies[bodyIdA];
    const bodyB = world.bodies[bodyIdB];
    const colorIndex = assignColor(graph, bodyIdA, bodyIdB, bodyA.type, bodyB.type);

    const isScalar =
        (contact.flags & ContactFlags.simMeshContact) !== 0 || colorIndex === OVERFLOW_INDEX;

    const color = graph.colors[colorIndex];
    contact.colorIndex = colorIndex;
    contact.localIndex = isScalar ? color.contacts.length : color.convexContacts.length;
    // Refresh the awake-column indices as the contact enters the graph (both bodies are awake here, their
    // localIndex current); thereafter maintained on each awake-body localIndex change.
    contact.bodySimIndexA = bodyA.type === BodyType.Static ? NULL_INDEX : bodyA.localIndex;
    contact.bodySimIndexB = bodyB.type === BodyType.Static ? NULL_INDEX : bodyB.localIndex;

    if (isScalar) {
        color.contacts.push({
            contactId: contact.contactId,
            manifoldStart: 0,
            manifoldCount: contact.manifoldCount,
        });
    } else {
        color.convexContacts.push(contact.contactId);
    }
}

// Remove a touching contact from its graph color (b3RemoveContactFromGraph). Takes the color/local
// index explicitly because the stopped-touching path re-homes the contact (overwriting its
// localIndex) before removing it from the graph; `meshContact` selects the scalar vs convex array.
export function removeContactFromGraph(
    world: WorldState,
    bodyIdA: number,
    bodyIdB: number,
    colorIndex: number,
    localIndex: number,
    meshContact: boolean,
): void {
    const color = world.constraintGraph.colors[colorIndex];

    if (colorIndex !== OVERFLOW_INDEX) {
        // May clear a static body's bit, which has no effect.
        clearBit(color.bodySet, bodyIdA);
        clearBit(color.bodySet, bodyIdB);
    }

    if (meshContact || colorIndex === OVERFLOW_INDEX) {
        const movedIndex = swapRemove(color.contacts, localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedContactId = color.contacts[localIndex].contactId;
            world.contacts[movedContactId].localIndex = localIndex;
        }
    } else {
        const movedIndex = swapRemove(color.convexContacts, localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedContactId = color.convexContacts[localIndex];
            world.contacts[movedContactId].localIndex = localIndex;
        }
    }
}

/** Clone a joint sim into its assigned color while awake (b3CreateJointInGraph).
 * @returns the fresh (zeroed) sim to fill. */
export function createJointInGraph(world: WorldState, joint: Joint): JointSim {
    const graph = world.constraintGraph;
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];
    const colorIndex = assignColor(
        graph,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
        bodyA.type,
        bodyB.type,
    );

    const color = graph.colors[colorIndex];
    const sim = emptyJointSim();
    color.jointSims.push(sim);
    joint.colorIndex = colorIndex;
    joint.localIndex = color.jointSims.length - 1;
    return sim;
}

// Re-home an existing joint sim into its assigned color (b3AddJointToGraph, used by wake/transfer).
// The port moves the sim object itself (the source array element is dropped), preserving its impulses.
export function addJointToGraph(world: WorldState, jointSim: JointSim, joint: Joint): void {
    const graph = world.constraintGraph;
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];
    const colorIndex = assignColor(
        graph,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
        bodyA.type,
        bodyB.type,
    );

    const color = graph.colors[colorIndex];
    color.jointSims.push(jointSim);
    joint.colorIndex = colorIndex;
    joint.localIndex = color.jointSims.length - 1;
}

// Remove a joint from its graph color (b3RemoveJointFromGraph).
export function removeJointFromGraph(
    world: WorldState,
    bodyIdA: number,
    bodyIdB: number,
    colorIndex: number,
    localIndex: number,
): void {
    const color = world.constraintGraph.colors[colorIndex];

    if (colorIndex !== OVERFLOW_INDEX) {
        // May clear a static body's bit, which has no effect.
        clearBit(color.bodySet, bodyIdA);
        clearBit(color.bodySet, bodyIdB);
    }

    const movedIndex = swapRemove(color.jointSims, localIndex);
    if (movedIndex !== NULL_INDEX) {
        const movedJoint = world.joints[color.jointSims[localIndex].jointId];
        movedJoint.localIndex = localIndex;
    }
}

/** Move a sleeping set's touching contacts and joints into the constraint graph (part of
 * b3WakeSolverSet). A sleeping set holds only touching contacts, so every one re-enters the graph. */
export function wakeSetConstraints(world: WorldState, set: SolverSet): void {
    for (let i = 0; i < set.contactIndices.length; ++i) {
        const contact = world.contacts[set.contactIndices[i]];
        addContactToGraph(world, contact);
        contact.setIndex = SetType.Awake;
    }

    for (let i = 0; i < set.jointSims.length; ++i) {
        const jointSim = set.jointSims[i];
        const joint = world.joints[jointSim.jointId];
        addJointToGraph(world, jointSim, joint);
        joint.setIndex = SetType.Awake;
    }
}
