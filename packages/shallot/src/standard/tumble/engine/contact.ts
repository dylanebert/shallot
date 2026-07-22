// Contacts: the persistent interaction between two shapes. Ported from Box3D's contact.c (Erin
// Catto, MIT). A contact threads a doubly-linked edge list through each of its two bodies and is
// keyed (contactId << 1 | edgeIndex). Contacts are born non-touching; touching is discovered during
// the step, which links islands and moves the contact into the constraint graph.
//
// Stage 7 (lifecycle) ports the create/destroy/register machinery and the deterministic shape
// ordering. No contact is actually created before the solver stage (pair finding runs in step), so
// the manifold + graph paths here are seamed to the solver stage. fround discipline per the README.

import { NULL_INDEX, swapRemove } from "./array";
import { type Body, BodyFlags, wakeBody } from "./body";
import { type CompoundData, getCompoundChild } from "./compound";
import { SetType } from "./core";
import { emptyCache, type SimplexCache } from "./distance";
import type { Capsule, Sphere } from "./geometry";
import { removeContactFromGraph } from "./graph";
import { allocId, freeId } from "./ids";
import { unlinkContact } from "./island";
import { emptySATCache, type SATCache } from "./manifold";
import type { AABB } from "./math";
import { maxf, type Quat, quat, type Transform, type Vec3 } from "./math";
import { getShapeMaterials, type Shape } from "./shape";
import { addKey, removeKey } from "./table";
import { BodyType, ShapeType } from "./types";
import type { WorldState } from "./world";

/** Which per-step collide list an awake contact belongs to (incremental partition, maintained here +
 * in solverset.ts against the create/destroy/wake/sleep event set; consumed by collide.ts). A contact
 * is enumerated by collide iff its `setIndex` is Awake, and then splits by recycle eligibility. */
export const AwakeContact = { None: 0, Recycle: 1, Other: 2 } as const;

/** Contact flags (b3ContactFlags): a persistent bank (low bits) and a per-step sim bank (0x10000+). */
export const ContactFlags = {
    contactTouchingFlag: 0x00000001,
    contactHitEventFlag: 0x00000002,
    contactEnableContactEvents: 0x00000004,
    contactStaticFlag: 0x00000008,
    contactRecycleFlag: 0x00000010,
    simTouchingFlag: 0x00010000,
    simDisjoint: 0x00020000,
    simStartedTouching: 0x00040000,
    simStoppedTouching: 0x00080000,
    simEnableHitEvent: 0x00100000,
    simEnablePreSolveEvents: 0x00200000,
    simMeshContact: 0x00400000,
    relativeTransformValid: 0x00800000,
} as const;

/** One end of a contact in a body's doubly-linked contact list (b3ContactEdge). */
export type ContactEdge = { bodyId: number; prevKey: number; nextKey: number };

/**
 * A single contact point in a persistent manifold (b3ManifoldPoint). Anchors are relative to each
 * body's center of mass in world space; the solver reads/writes the impulses across sub-steps.
 */
export type ManifoldPoint = {
    anchorA: Vec3;
    anchorB: Vec3;
    separation: number;
    baseSeparation: number;
    normalImpulse: number;
    totalNormalImpulse: number;
    normalVelocity: number;
    featureId: number;
    triangleIndex: number;
    persisted: boolean;
};

/** The persistent contact manifold between two shapes (b3Manifold). 1–4 points. */
export type Manifold = {
    points: ManifoldPoint[];
    normal: Vec3;
    twistImpulse: number;
    frictionImpulse: Vec3;
    rollingImpulse: Vec3;
    pointCount: number;
};

/** GJK/SAT warm-start caches carried on a convex contact (b3ContactCache). */
export type ConvexContactCache = { simplexCache: SimplexCache; satCache: SATCache };

/** Per-triangle warm-start cache for a mesh/height-field contact (b3TriangleCache). */
export type TriangleCache = { triangleIndex: number; cache: ConvexContactCache };

/**
 * State a mesh/height-field contact carries in place of the convex cache (b3MeshContact): the
 * per-triangle caches from the last narrowphase pass and the world-space bounds those triangles
 * were queried against, so an unmoved body can skip re-querying the BVH.
 */
export type MeshContact = { triangleCache: TriangleCache[]; queryBounds: AABB };

/** The persistent interaction between two shapes (b3Contact). */
export type Contact = {
    setIndex: number;
    colorIndex: number;
    localIndex: number;
    edges: [ContactEdge, ContactEdge];
    shapeIdA: number;
    shapeIdB: number;
    childIndex: number;
    islandId: number;
    islandIndex: number;
    contactId: number;
    // The two bodies' awake-column indices (localIndex, or NULL_INDEX for a static side), the slot the
    // kernel gathers each body's sim/state through. Stored (not recomputed at read): maintained at contact
    // create + graph add + every awake-body localIndex change (contact.ts / solverset.ts / body.ts), so the
    // solver's writeRow and the recycle dispatch read a warm field instead of re-loading the body records.
    bodySimIndexA: number;
    bodySimIndexB: number;
    flags: number;
    // Recycle eligibility's per-contact-constant half (dynamic-dynamic direct-convex): !static && !mesh
    // && shapeA convex. Fixed at create; the only mutable input to eligibility is the two bodies'
    // setIndex. `collideKind`/`collideIndex` are this contact's live slot in the incremental collide
    // partition (world.awakeRecycleContacts / awakeOtherContacts) — AwakeContact.None when not awake.
    recycleStable: boolean;
    collideKind: number;
    collideIndex: number;
    // Manifold(s) computed by narrowphase during the step; the GJK/SAT cache persists across steps.
    manifolds: Manifold[];
    manifoldCount: number;
    // A convex contact uses `cache`; a mesh/height-field contact uses `meshContact` instead (a union
    // in C). Both are always allocated here; the narrowphase reads the one its shape type dictates.
    cache: ConvexContactCache;
    meshContact: MeshContact;
    // Cached relative pose for the contact-recycling test (updated each full narrowphase pass).
    cachedRotationA: Quat;
    cachedRotationB: Quat;
    cachedRelativePose: Transform;
    friction: number;
    restitution: number;
    rollingResistance: number;
    tangentVelocity: Vec3;
    generation: number;
};

function makeContact(generation: number): Contact {
    return {
        setIndex: NULL_INDEX,
        colorIndex: NULL_INDEX,
        localIndex: NULL_INDEX,
        edges: [
            { bodyId: NULL_INDEX, prevKey: NULL_INDEX, nextKey: NULL_INDEX },
            { bodyId: NULL_INDEX, prevKey: NULL_INDEX, nextKey: NULL_INDEX },
        ],
        shapeIdA: NULL_INDEX,
        shapeIdB: NULL_INDEX,
        childIndex: 0,
        islandId: NULL_INDEX,
        islandIndex: NULL_INDEX,
        contactId: NULL_INDEX,
        bodySimIndexA: NULL_INDEX,
        bodySimIndexB: NULL_INDEX,
        flags: 0,
        recycleStable: false,
        collideKind: AwakeContact.None,
        collideIndex: NULL_INDEX,
        manifolds: [],
        manifoldCount: 0,
        cache: { simplexCache: emptyCache(), satCache: emptySATCache() },
        meshContact: {
            triangleCache: [],
            queryBounds: {
                lowerBound: { x: 0, y: 0, z: 0 },
                upperBound: { x: 0, y: 0, z: 0 },
            },
        },
        cachedRotationA: quat.identity(),
        cachedRotationB: quat.identity(),
        cachedRelativePose: { p: { x: 0, y: 0, z: 0 }, q: quat.identity() },
        friction: 0,
        restitution: 0,
        rollingResistance: 0,
        tangentVelocity: { x: 0, y: 0, z: 0 },
        generation,
    };
}

// The shape-pair dispatch table (s_registers): which ordered pairs collide and which order is
// canonical (primary). Built once, matching b3InitializeContactRegisters. Determinism depends on
// the canonical A/B order coming from shape types, not creation order.
const SUPPORTED: boolean[][] = [];
const PRIMARY: boolean[][] = [];
let registersInitialized = false;

function addType(type1: number, type2: number): void {
    SUPPORTED[type1][type2] = true;
    PRIMARY[type1][type2] = true;
    if (type1 !== type2) {
        SUPPORTED[type2][type1] = true;
        PRIMARY[type2][type1] = false;
    }
}

export function initializeContactRegisters(): void {
    if (registersInitialized) {
        return;
    }
    for (let i = 0; i < 6; ++i) {
        SUPPORTED[i] = [false, false, false, false, false, false];
        PRIMARY[i] = [false, false, false, false, false, false];
    }
    addType(ShapeType.Sphere, ShapeType.Sphere);
    addType(ShapeType.Capsule, ShapeType.Sphere);
    addType(ShapeType.Capsule, ShapeType.Capsule);
    addType(ShapeType.Compound, ShapeType.Sphere);
    addType(ShapeType.Compound, ShapeType.Capsule);
    addType(ShapeType.Compound, ShapeType.Hull);
    addType(ShapeType.Hull, ShapeType.Sphere);
    addType(ShapeType.Hull, ShapeType.Capsule);
    addType(ShapeType.Hull, ShapeType.Hull);
    addType(ShapeType.Mesh, ShapeType.Sphere);
    addType(ShapeType.Mesh, ShapeType.Capsule);
    addType(ShapeType.Mesh, ShapeType.Hull);
    addType(ShapeType.HeightField, ShapeType.Sphere);
    addType(ShapeType.HeightField, ShapeType.Capsule);
    addType(ShapeType.HeightField, ShapeType.Hull);
    registersInitialized = true;
}

// Which collide list an awake contact belongs to. Enumerated iff setIndex is Awake; then recycle iff
// its per-contact-constant `recycleStable` holds and both bodies are awake-resident (the kernel recycle
// pass indexes the resident columns by localIndex, so both must be in the awake set). Order-free — the
// per-contact narrowphase/recycle work is independent and stateChanges is sorted before acting.
function classifyAwakeContact(world: WorldState, contact: Contact): number {
    if (contact.setIndex !== SetType.Awake) {
        return AwakeContact.None;
    }
    if (contact.recycleStable) {
        const bodyA = world.bodies[contact.edges[0].bodyId];
        const bodyB = world.bodies[contact.edges[1].bodyId];
        if (bodyA.setIndex === SetType.Awake && bodyB.setIndex === SetType.Awake) {
            return AwakeContact.Recycle;
        }
    }
    return AwakeContact.Other;
}

// Unlink a contact from its current collide list via swap-remove (leaves collideKind/Index for the
// caller to reset). The freed slot inherits the list's tail; fix the moved contact's cached index.
function detachAwakeContact(world: WorldState, contact: Contact): void {
    const kind = contact.collideKind;
    if (kind === AwakeContact.None) {
        return;
    }
    const list =
        kind === AwakeContact.Recycle ? world.awakeRecycleContacts : world.awakeOtherContacts;
    const idx = contact.collideIndex;
    if (swapRemove(list, idx) !== NULL_INDEX) {
        world.contacts[list[idx]].collideIndex = idx;
    }
}

/** Move a contact to the collide list its current state dictates (create, or a body/contact setIndex
 * change). Idempotent: re-running from either endpoint of a two-body event converges to the same slot. */
export function updateAwakeContact(world: WorldState, contact: Contact): void {
    const target = classifyAwakeContact(world, contact);
    if (target === contact.collideKind) {
        return;
    }
    detachAwakeContact(world, contact);
    if (target === AwakeContact.None) {
        contact.collideKind = AwakeContact.None;
        contact.collideIndex = NULL_INDEX;
    } else {
        const list =
            target === AwakeContact.Recycle ? world.awakeRecycleContacts : world.awakeOtherContacts;
        contact.collideIndex = list.length;
        list.push(contact.contactId);
        contact.collideKind = target;
    }
}

// Force a contact out of the collide partition (destroy path — the contact is going away, so classify
// would wrongly keep an awake one).
function removeAwakeContact(world: WorldState, contact: Contact): void {
    if (contact.collideKind === AwakeContact.None) {
        return;
    }
    detachAwakeContact(world, contact);
    contact.collideKind = AwakeContact.None;
    contact.collideIndex = NULL_INDEX;
}

/** Rewrite `bodySimIndexA/B` on this body's side of each of its contacts (the awake-column index the
 * solver + recycle pass gather through). Called when an *awake* body's localIndex changes without its
 * set membership changing — the swap-remove that migrates a surviving awake body into a freed slot. */
export function writeBodySimIndex(world: WorldState, body: Body): void {
    const simIndex = body.type === BodyType.Static ? NULL_INDEX : body.localIndex;
    let contactKey = body.headContactKey;
    while (contactKey !== NULL_INDEX) {
        const edgeIndex = contactKey & 1;
        const contact = world.contacts[contactKey >> 1];
        contactKey = contact.edges[edgeIndex].nextKey;
        if (edgeIndex === 0) {
            contact.bodySimIndexA = simIndex;
        } else {
            contact.bodySimIndexB = simIndex;
        }
    }
}

/** Re-partition every contact on a body's edge list after the body's setIndex changed (wake/sleep/
 * transfer), and refresh this body's `bodySimIndex` side (localIndex was reassigned in the same move).
 * Walks the doubly-linked contact list; each contact is reclassified against current state. */
export function reclassifyBodyContacts(world: WorldState, body: Body): void {
    const simIndex = body.type === BodyType.Static ? NULL_INDEX : body.localIndex;
    let contactKey = body.headContactKey;
    while (contactKey !== NULL_INDEX) {
        const edgeIndex = contactKey & 1;
        const contact = world.contacts[contactKey >> 1];
        contactKey = contact.edges[edgeIndex].nextKey;
        if (edgeIndex === 0) {
            contact.bodySimIndexA = simIndex;
        } else {
            contact.bodySimIndexB = simIndex;
        }
        updateAwakeContact(world, contact);
    }
}

export function createContact(
    world: WorldState,
    shapeA: Shape,
    shapeB: Shape,
    childIndex: number,
): void {
    const typeA = shapeA.type;
    const typeB = shapeB.type;

    if (SUPPORTED[typeA][typeB] === false) {
        // For example, no mesh vs mesh collision
        return;
    }

    if (PRIMARY[typeA][typeB] === false) {
        // flip order to the canonical (primary) direction
        createContact(world, shapeB, shapeA, childIndex);
        return;
    }

    const bodyA = world.bodies[shapeA.bodyId];
    const bodyB = world.bodies[shapeB.bodyId];

    let setIndex: number;
    if (bodyA.setIndex === SetType.Awake || bodyB.setIndex === SetType.Awake) {
        setIndex = SetType.Awake;
    } else {
        // sleeping and non-touching contacts live in the disabled set until found touching
        setIndex = SetType.Disabled;
    }

    const set = world.solverSets[setIndex];

    const contactId = allocId(world.contactIdPool);
    if (contactId === world.contacts.length) {
        world.contacts.push(makeContact(0));
    }

    const shapeIdA = shapeA.id;
    const shapeIdB = shapeB.id;

    const generation = world.contacts[contactId].generation;
    const contact = makeContact(generation + 1);
    world.contacts[contactId] = contact;
    contact.contactId = contactId;
    contact.setIndex = setIndex;
    contact.localIndex = set.contactIndices.length;
    contact.shapeIdA = shapeIdA;
    contact.shapeIdB = shapeIdB;
    contact.childIndex = childIndex;

    // Both bodies must enable recycling
    if (
        (bodyA.flags & BodyFlags.enableContactRecycling) !== 0 &&
        (bodyB.flags & BodyFlags.enableContactRecycling) !== 0
    ) {
        contact.flags |= ContactFlags.contactRecycleFlag;
    }

    if (shapeA.type === ShapeType.Mesh || shapeA.type === ShapeType.HeightField) {
        contact.flags |= ContactFlags.simMeshContact;
    } else if (shapeA.type === ShapeType.Compound) {
        // A compound whose selected child is a mesh drives the mesh-contact solver path.
        const child = getCompoundChild(shapeA.compound as CompoundData, childIndex);
        if (child.type === ShapeType.Mesh) {
            contact.flags |= ContactFlags.simMeshContact;
        }
    }

    if (bodyA.type === BodyType.Static || bodyB.type === BodyType.Static) {
        contact.flags |= ContactFlags.contactStaticFlag;
    }

    if (shapeA.enableContactEvents || shapeB.enableContactEvents) {
        contact.flags |= ContactFlags.contactEnableContactEvents;
    }

    // Connect to body A
    {
        contact.edges[0].bodyId = shapeA.bodyId;
        contact.edges[0].prevKey = NULL_INDEX;
        contact.edges[0].nextKey = bodyA.headContactKey;

        const keyA = (contactId << 1) | 0;
        const headContactKey = bodyA.headContactKey;
        if (headContactKey !== NULL_INDEX) {
            const headContact = world.contacts[headContactKey >> 1];
            headContact.edges[headContactKey & 1].prevKey = keyA;
        }
        bodyA.headContactKey = keyA;
        bodyA.contactCount += 1;
    }

    // Connect to body B
    {
        contact.edges[1].bodyId = shapeB.bodyId;
        contact.edges[1].prevKey = NULL_INDEX;
        contact.edges[1].nextKey = bodyB.headContactKey;

        const keyB = (contactId << 1) | 1;
        const headContactKey = bodyB.headContactKey;
        if (headContactKey !== NULL_INDEX) {
            const headContact = world.contacts[headContactKey >> 1];
            headContact.edges[headContactKey & 1].prevKey = keyB;
        }
        bodyB.headContactKey = keyB;
        bodyB.contactCount += 1;
    }

    // Track the contact's persistent manifold directory slot (the block is allocated on first touch).
    world.manifoldStore.ensureSlot(contactId);

    // Add to pair set for fast lookup
    addKey(world.broadPhase.pairSet, shapeIdA, shapeIdB, childIndex);

    // Contacts are created non-touching.
    set.contactIndices.push(contactId);

    let radiusA = 0;
    if (typeA === ShapeType.Sphere) {
        radiusA = (shapeA.sphere as Sphere).radius;
    } else if (typeA === ShapeType.Capsule) {
        radiusA = (shapeA.capsule as Capsule).radius;
    }

    let radiusB = 0;
    if (typeB === ShapeType.Sphere) {
        radiusB = (shapeB.sphere as Sphere).radius;
    } else if (typeB === ShapeType.Capsule) {
        radiusB = (shapeB.capsule as Capsule).radius;
    }

    const maxRadius = maxf(radiusA, radiusB);

    contact.rollingResistance =
        maxf(
            getShapeMaterials(shapeA)[0].rollingResistance,
            getShapeMaterials(shapeB)[0].rollingResistance,
        ) * maxRadius;

    if (shapeA.enablePreSolveEvents || shapeB.enablePreSolveEvents) {
        contact.flags |= ContactFlags.simEnablePreSolveEvents;
    }

    // Recycle eligibility's constant half (flags + canonical shapeA type — both fixed for this contact's
    // life). Enter the incremental collide partition. `typeA` is the primary/canonical A after the flip.
    contact.recycleStable =
        (contact.flags & ContactFlags.contactStaticFlag) === 0 &&
        (contact.flags & ContactFlags.simMeshContact) === 0 &&
        (typeA === ShapeType.Sphere || typeA === ShapeType.Capsule || typeA === ShapeType.Hull);
    updateAwakeContact(world, contact);

    // Seed the awake-column indices for the solver + recycle readers (thereafter maintained at graph add
    // + every awake-body localIndex change). A sleeping-body side seeds its sleeping-set index — unread
    // until that body wakes (reclassifyBodyContacts refreshes it) and the contact enters a solved/recycle
    // path. Static side is NULL.
    contact.bodySimIndexA = bodyA.type === BodyType.Static ? NULL_INDEX : bodyA.localIndex;
    contact.bodySimIndexB = bodyB.type === BodyType.Static ? NULL_INDEX : bodyB.localIndex;
}

// A contact is destroyed when proxies stop overlapping, a body/shape is destroyed or disabled, a
// body changes type, or a filter changes.
export function destroyContact(world: WorldState, contact: Contact, wakeBodies: boolean): void {
    removeAwakeContact(world, contact);
    removeKey(world.broadPhase.pairSet, contact.shapeIdA, contact.shapeIdB, contact.childIndex);

    // Drop the manifold views and release the persistent slot — free the manifold block, zero the block
    // descriptor, and cold the convex GJK/SAT cache so a recycled contactId doesn't inherit stale
    // separating-axis indices from this pair's hulls (b3FreeManifolds + the contact's cache going away).
    contact.manifolds = [];
    contact.manifoldCount = 0;
    world.manifoldStore.freeSlot(contact.contactId);

    const edgeA = contact.edges[0];
    const edgeB = contact.edges[1];

    const bodyIdA = edgeA.bodyId;
    const bodyIdB = edgeB.bodyId;
    const bodyA = world.bodies[bodyIdA];
    const bodyB = world.bodies[bodyIdB];

    const flags = contact.flags;
    const touching = (flags & ContactFlags.contactTouchingFlag) !== 0;

    // End touch event
    if (touching && (flags & ContactFlags.contactEnableContactEvents) !== 0) {
        const worldId = world.worldId;
        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        world.contactEndEvents[world.endEventArrayIndex].push({
            shapeIdA: { index1: shapeA.id + 1, world0: worldId, generation: shapeA.generation },
            shapeIdB: { index1: shapeB.id + 1, world0: worldId, generation: shapeB.generation },
            contactId: {
                index1: contact.contactId + 1,
                world0: worldId,
                generation: contact.generation,
            },
        });
    }

    // Remove from body A's list
    if (edgeA.prevKey !== NULL_INDEX) {
        const prevContact = world.contacts[edgeA.prevKey >> 1];
        prevContact.edges[edgeA.prevKey & 1].nextKey = edgeA.nextKey;
    }
    if (edgeA.nextKey !== NULL_INDEX) {
        const nextContact = world.contacts[edgeA.nextKey >> 1];
        nextContact.edges[edgeA.nextKey & 1].prevKey = edgeA.prevKey;
    }

    const contactId = contact.contactId;

    const edgeKeyA = (contactId << 1) | 0;
    if (bodyA.headContactKey === edgeKeyA) {
        bodyA.headContactKey = edgeA.nextKey;
    }
    bodyA.contactCount -= 1;

    // Remove from body B's list
    if (edgeB.prevKey !== NULL_INDEX) {
        const prevContact = world.contacts[edgeB.prevKey >> 1];
        prevContact.edges[edgeB.prevKey & 1].nextKey = edgeB.nextKey;
    }
    if (edgeB.nextKey !== NULL_INDEX) {
        const nextContact = world.contacts[edgeB.nextKey >> 1];
        nextContact.edges[edgeB.nextKey & 1].prevKey = edgeB.prevKey;
    }

    const edgeKeyB = (contactId << 1) | 1;
    if (bodyB.headContactKey === edgeKeyB) {
        bodyB.headContactKey = edgeB.nextKey;
    }
    bodyB.contactCount -= 1;

    // A mesh contact's triangle cache is plain JS objects; GC reclaims it (b3Array_Destroy in C).
    if ((flags & ContactFlags.simMeshContact) !== 0) {
        contact.meshContact.triangleCache = [];
    }

    // Remove contact from the array that owns it
    if (contact.islandId !== NULL_INDEX) {
        unlinkContact(world, contact);
    }

    if (contact.colorIndex !== NULL_INDEX) {
        // contact is an active constraint in the graph
        removeContactFromGraph(
            world,
            contact.edges[0].bodyId,
            contact.edges[1].bodyId,
            contact.colorIndex,
            contact.localIndex,
            (contact.flags & ContactFlags.simMeshContact) !== 0,
        );
    } else {
        // contact is non-touching, sleeping, or a sensor
        const set = world.solverSets[contact.setIndex];
        const localIndex = contact.localIndex;
        const movedIndex = swapRemove(set.contactIndices, localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedContactIndex = set.contactIndices[localIndex];
            world.contacts[movedContactIndex].localIndex = localIndex;
        }
    }

    // Free contact and id (preserve generation)
    contact.contactId = NULL_INDEX;
    contact.setIndex = NULL_INDEX;
    contact.colorIndex = NULL_INDEX;
    contact.localIndex = NULL_INDEX;
    freeId(world.contactIdPool, contactId);

    if (wakeBodies && touching) {
        wakeBody(world, bodyA);
        wakeBody(world, bodyB);
    }
}
