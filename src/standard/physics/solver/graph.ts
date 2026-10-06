import { ContactField, contactField, setContactField } from "../collision/contact";
import { bodyType } from "../kernel/filtercolumns";
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
// The transient per-step constraint arrays live in wasm columns, not here.
//
// Coloring is integer-only, so no fround discipline applies here.

import { ContactFlags } from "../collision/contact";
import { NULL_INDEX, swapRemove } from "../common/array";
import {
    type BitSet,
    clearBit,
    createBitSet,
    getBit,
    setBitCountAndClear,
    setBitGrow,
} from "../common/bitset";
import {
    DYNAMIC_COLOR_COUNT,
    GRAPH_COLOR_COUNT,
    OVERFLOW_INDEX,
    SetType,
} from "../common/constants";
import { BodyType } from "../common/types";
import {
    appendJointRecord,
    jointArrayCount,
    jointAt,
    moveJointRecord,
    removeJointRecord,
} from "../kernel/jointcolumns";
import type { SolverSet } from "../world/solverset";
import type { WorldState } from "../world/world";
import type { Joint } from "./joint";

/** One touching contact's entry in a graph color (b3ContactSpec). */
export type ContactSpec = { contactId: number; manifoldStart: number; manifoldCount: number };

/** One solver color: the constraints solved together (b3GraphColor). `bodySet` is unused on the
 * overflow color. */
export type GraphColor = {
    bodySet: BitSet;
    contacts: ContactSpec[];
    convexContacts: number[];
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
        colors.push({ bodySet, contacts: [], convexContacts: [] });
    }
    return { colors };
}

/** Canonical color assignment shared by b3AddContactToGraph / b3AssignJointColor. */
function assignColor(
    graph: ConstraintGraph,
    bodyIdA: number,
    bodyIdB: number,
    typeA: BodyType,
    typeB: BodyType,
): number {
    return greedyColor(graph, bodyIdA, bodyIdB, typeA, typeB);
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
export function addContactToGraph(world: WorldState, contact: number): void {
    const graph = world.constraintGraph;

    const bodyIdA = contactField(world, contact, ContactField.bodyIdA + 3 * 0);
    const bodyIdB = contactField(world, contact, ContactField.bodyIdA + 3 * 1);
    const bodyA = world.bodies[bodyIdA];
    const bodyB = world.bodies[bodyIdB];
    const colorIndex = assignColor(
        graph,
        bodyIdA,
        bodyIdB,
        bodyType(world, bodyA.id),
        bodyType(world, bodyB.id),
    );

    const isScalar =
        (contactField(world, contact, ContactField.flags) & ContactFlags.simMeshContact) !== 0 ||
        colorIndex === OVERFLOW_INDEX;

    const color = graph.colors[colorIndex];
    setContactField(world, contact, ContactField.colorIndex, colorIndex);
    setContactField(
        world,
        contact,
        ContactField.localIndex,
        isScalar ? color.contacts.length : color.convexContacts.length,
    );
    // Refresh the awake-column indices as the contact enters the graph (both bodies are awake here, their
    // localIndex current); thereafter maintained on each awake-body localIndex change.
    setContactField(
        world,
        contact,
        ContactField.bodySimIndexA,
        bodyType(world, bodyA.id) === BodyType.Static ? NULL_INDEX : bodyA.localIndex,
    );
    setContactField(
        world,
        contact,
        ContactField.bodySimIndexB,
        bodyType(world, bodyB.id) === BodyType.Static ? NULL_INDEX : bodyB.localIndex,
    );

    if (isScalar) {
        color.contacts.push({
            contactId: contact,
            manifoldStart: 0,
            manifoldCount: contactField(world, contact, ContactField.manifoldCount),
        });
    } else {
        color.convexContacts.push(contact);
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
            setContactField(world, movedContactId, ContactField.localIndex, localIndex);
        }
    } else {
        const movedIndex = swapRemove(color.convexContacts, localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedContactId = color.convexContacts[localIndex];
            setContactField(world, movedContactId, ContactField.localIndex, localIndex);
        }
    }
}

/** Allocate a zeroed sim in the joint's assigned graph color (b3CreateJointInGraph). */
export function createJointInGraph(world: WorldState, joint: Joint): void {
    const graph = world.constraintGraph;
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];
    const colorIndex = assignColor(
        graph,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
        bodyType(world, bodyA.id),
        bodyType(world, bodyB.id),
    );

    joint.colorIndex = colorIndex;
    joint.localIndex = appendJointRecord(world, colorIndex);
}

// Copy into the target color and swap-remove the source, as b3AddJointToGraph does.
export function addJointToGraph(world: WorldState, joint: Joint): void {
    const graph = world.constraintGraph;
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];
    const colorIndex = assignColor(
        graph,
        joint.edges[0].bodyId,
        joint.edges[1].bodyId,
        bodyType(world, bodyA.id),
        bodyType(world, bodyB.id),
    );

    const destination = moveJointRecord(world, joint, colorIndex);
    joint.colorIndex = colorIndex;
    joint.localIndex = destination;
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

    removeJointRecord(world, colorIndex, localIndex);
}

/** Move a sleeping set's touching contacts and joints into the constraint graph (part of
 * b3WakeSolverSet). A sleeping set holds only touching contacts, so every one re-enters the graph. */
export function wakeSetConstraints(world: WorldState, set: SolverSet): void {
    for (let i = 0; i < set.contactIndices.length; ++i) {
        const contact = set.contactIndices[i];
        addContactToGraph(world, contact);
        setContactField(world, contact, ContactField.setIndex, SetType.Awake);
    }

    const key = GRAPH_COLOR_COUNT + set.setIndex;
    const count = jointArrayCount(world, key);
    for (let i = 0; i < count; ++i) {
        // Swap-removal parks the original tail in the consumed prefix; visit original order.
        const joint = jointAt(world, key, Math.min(i, count - 1 - i));
        addJointToGraph(world, joint);
        joint.setIndex = SetType.Awake;
    }
}
