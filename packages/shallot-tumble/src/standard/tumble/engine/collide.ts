// Narrow-phase — Box3D's b3Collide (physics_world.c) + the convex-contact manifold bridge
// (contact.c), Erin Catto, MIT. For every awake contact this recomputes (or recycles) the manifold,
// tracks touching-state transitions, and re-homes contacts between the non-touching set and the
// constraint graph. The geometry manifolds come from the stage-6 collide functions; this layer maps
// them into the persistent, center-of-mass-relative b3Manifold the solver reads.
//
// Convex + mesh + height-field + compound contacts. Started touching
// links the contact into an island (b3LinkContact, waking a sleeping partner); stopped touching
// unlinks it. Every op is fround-wrapped; see the README.

import { NULL_INDEX, swapRemove } from "./array";
import { BodyFlags, getBodySim } from "./body";
import {
    D_CONTACT,
    D_GEOM_A,
    D_GEOM_B,
    D_TYPE_A,
    D_TYPE_B,
    D_XF_A,
    D_XF_B,
    DISPATCH_STRIDE,
    R_BITS,
    R_CONTACT,
    R_ELIGIBLE,
    R_LOCAL_A,
    R_LOCAL_B,
    R_SHAPE_A,
    R_SHAPE_B,
    R_WAS_TOUCHING,
    RECYCLE_STRIDE,
} from "./columns";
import { type ChildShape, type CompoundData, getCompoundChild } from "./compound";
import { type Contact, ContactFlags, destroyContact, type Manifold } from "./contact";
import type { StepContext } from "./contactsolver";
import { CONTACT_RECYCLE_ANGULAR_DISTANCE, SetType, SPECULATIVE_DISTANCE } from "./core";
import { rebuildGeometry } from "./geocolumns";
import type { Capsule, Sphere } from "./geometry";
import { addContactToGraph, removeContactFromGraph } from "./graph";
import type { HullData } from "./hull";
import { linkContact, unlinkContact } from "./island";
import { kernel, ParKind, runPar } from "./kernel";
import {
    collideCapsuleAndSphere,
    collideCapsules,
    collideHullAndCapsule,
    collideHullAndSphere,
    collideHulls,
    collideSpheres,
    makeFeatureId,
    makeLocalManifold,
} from "./manifold";
import type { ManifoldStore } from "./manifoldstore";
import {
    aabb,
    f32,
    invMulWorldTransforms,
    invMulWorldTransformsOut,
    mat3,
    minf,
    mulWorldTransforms,
    type Quat,
    quat,
    subPos,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
} from "./math";
import { computeMeshManifolds } from "./mesh_contact";
import { getShapeMaterial, type Shape } from "./shape";
import { BodyType, ShapeType } from "./types";
import type { WorldState } from "./world";

const UINT32_MAX = 0xffffffff;

// Reused narrow-phase scratch for the convex-contact manifold bridge. The collide functions treat a
// LocalManifold as a caller-owned buffer (write points, set pointCount), and computeConvexManifold
// copies everything it needs out into the persistent manifold before returning, so a single
// module-scope buffer serves every convex contact — it is never live across a re-entrant call. Sized
// to the convex capacity (32). Avoids a fresh 32-slot manifold allocation per contact per step.
const CONVEX_CAPACITY = 32;
const scratchGeom = makeLocalManifold(CONVEX_CAPACITY);

// Reused scratch for computeConvexManifold's in-place rebuild. `matrixAScratch` holds xfA's rotation
// (fromQuatOut). `oldFeat`/`oldImp` snapshot the previous manifold's feature ids + impulses before
// the points are overwritten in place, so warm-start matching reads the snapshot (and marks claimed
// entries there) instead of the objects it is reusing. A convex manifold holds ≤ 4 points; sized to
// CONVEX_CAPACITY for headroom.
const matrixAScratch = mat3.zero();
const oldFeat: number[] = new Array(CONVEX_CAPACITY);
const oldImp: number[] = new Array(CONVEX_CAPACITY);

// Per-step reused buffers + per-call scratch for the collect/recycle/finish loops. Module scope is
// safe: collide() never re-enters (one thread, worlds step sequentially) and nothing here is live
// across calls — the arrays reset on entry, the scratch within one contact's processing.
const stateChangeIds: number[] = [];
// The batched kernel recycle pass runs over `world.awakeRecycleContacts` (the incremental partition) and
// the non-recycle walk over `world.awakeOtherContacts`; neither is gathered here anymore.
// The kernel recycle pass's per-contact result, snapshotted out of the wasm output column before the
// needs-narrowphase collect (whose pool grow would detach/clobber the column). Reused, grow-only.
const recycleResults: number[] = [];
// Shared read-only zero: warm-start impulse resets (setters copy) and static-body max extents.
const zero: Vec3 = { x: 0, y: 0, z: 0 };
// The empty manifold set a not-touching contact holds. Shared — contact.manifolds is only ever
// reassigned whole (alloc / this), never mutated in place.
const NO_MANIFOLDS: Manifold[] = [];
const recycleXf: Transform = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
const recycleExtent: Vec3 = { x: 0, y: 0, z: 0 };
const recycleTmp: Vec3 = { x: 0, y: 0, z: 0 };
const recycleQr: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const recycleConj: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const recycleDq: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const recycleMatrixA = mat3.zero();
const recycleMatrixB = mat3.zero();
const recycleDc: Vec3 = { x: 0, y: 0, z: 0 };
const centerAScratch: Vec3 = { x: 0, y: 0, z: 0 };
const centerBScratch: Vec3 = { x: 0, y: 0, z: 0 };
const tangentAScratch: Vec3 = { x: 0, y: 0, z: 0 };
const tangentBScratch: Vec3 = { x: 0, y: 0, z: 0 };

// Compute the convex-convex manifold and map it into the contact's persistent manifold, carrying
// warm-start impulses forward by feature id (b3ComputeConvexManifold). The manifold is column-resident
// (world.manifoldStore); anchors/normal are written through the pool-backed view.
function computeConvexManifold(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    xfA: WorldTransform,
    shapeB: Shape,
    xfB: WorldTransform,
): boolean {
    const typeA = shapeA.type;
    const typeB = shapeB.type;
    const cache = contact.cache;

    const capacity = CONVEX_CAPACITY;
    const geom = scratchGeom;
    geom.pointCount = 0;
    const transformBtoA = invMulWorldTransforms(xfA, xfB);

    if (typeA === ShapeType.Sphere) {
        collideSpheres(
            geom,
            capacity,
            shapeA.sphere as Sphere,
            shapeB.sphere as Sphere,
            transformBtoA,
        );
    } else if (typeA === ShapeType.Capsule) {
        if (typeB === ShapeType.Sphere) {
            collideCapsuleAndSphere(
                geom,
                capacity,
                shapeA.capsule as Capsule,
                shapeB.sphere as Sphere,
                transformBtoA,
            );
        } else {
            collideCapsules(
                geom,
                capacity,
                shapeA.capsule as Capsule,
                shapeB.capsule as Capsule,
                transformBtoA,
            );
        }
    } else {
        // hull A
        if (typeB === ShapeType.Sphere) {
            collideHullAndSphere(
                geom,
                capacity,
                shapeA.hull as HullData,
                shapeB.sphere as Sphere,
                transformBtoA,
                cache.simplexCache,
            );
        } else if (typeB === ShapeType.Capsule) {
            collideHullAndCapsule(
                geom,
                capacity,
                shapeA.hull as HullData,
                shapeB.capsule as Capsule,
                transformBtoA,
                cache.simplexCache,
            );
        } else {
            collideHulls(
                geom,
                capacity,
                shapeA.hull as HullData,
                shapeB.hull as HullData,
                transformBtoA,
                cache.satCache,
            );
        }
    }

    if (geom.pointCount === 0) {
        contact.manifolds = NO_MANIFOLDS;
        contact.manifoldCount = 0;
        return false;
    }

    // A convex contact holds one manifold. On the non-touching → touching transition allocate a fresh
    // block (which may recycle a freed block still holding another contact's warm-start data) and zero
    // the friction/twist/rolling warm-start; on a still-touching step reuse the resident block in place
    // (its impulses are last step's, the warm-start the solver carries forward).
    const store = world.manifoldStore;
    let manifold: Manifold;
    let oldCount: number;
    if (contact.manifoldCount === 0) {
        contact.manifolds = store.alloc(contact.contactId, 1);
        contact.manifoldCount = 1;
        manifold = contact.manifolds[0];
        manifold.frictionImpulse = zero;
        manifold.rollingImpulse = zero;
        manifold.twistImpulse = 0;
        oldCount = 0;
    } else {
        manifold = contact.manifolds[0];
        oldCount = manifold.pointCount;
    }
    const points = manifold.points;

    // Snapshot the previous feature ids + impulses before overwriting the points in place; warm-start
    // matching (below) reads and claims the snapshot, never the reused points.
    for (let j = 0; j < oldCount; ++j) {
        oldFeat[j] = points[j].featureId;
        oldImp[j] = points[j].normalImpulse;
    }

    const n = geom.pointCount;
    manifold.pointCount = n;

    const matrixA = mat3.fromQuatOut(xfA.q, matrixAScratch);
    manifold.normal = mat3.mulV(matrixA, geom.normal);

    const offset = subPos(xfA.p, xfB.p);
    for (let i = 0; i < n; ++i) {
        const source = geom.points[i];
        const pt = points[i];
        pt.anchorA = mat3.mulV(matrixA, source.point);
        pt.anchorB = vec3.add(pt.anchorA, offset);
        pt.separation = source.separation;
        pt.baseSeparation = 0;
        pt.totalNormalImpulse = 0;
        pt.normalVelocity = 0;
        pt.featureId = makeFeatureId(source.pair);
        pt.triangleIndex = NULL_INDEX;
        pt.persisted = false;
    }

    // Carry impulses from any matching old point (by feature id) via the snapshot.
    for (let i = 0; i < n; ++i) {
        const pt2 = points[i];
        for (let j = 0; j < oldCount; ++j) {
            if (pt2.featureId === oldFeat[j]) {
                pt2.normalImpulse = oldImp[j];
                pt2.persisted = true;
                oldFeat[j] = UINT32_MAX; // claimed
                break;
            }
        }
        if (pt2.persisted === false) {
            pt2.normalImpulse = 0;
        }
    }

    return true;
}

// Update a convex-vs-convex contact: manifold + material mixing + tangent velocity (b3UpdateConvexContact).
function updateConvexContact(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    xfA: WorldTransform,
    shapeB: Shape,
    xfB: WorldTransform,
    flip: boolean,
): boolean {
    const touching = computeConvexManifold(world, contact, shapeA, xfA, shapeB, xfB);
    if (touching === false) {
        return false;
    }

    if (flip) {
        const manifold = contact.manifolds[0];
        manifold.normal = vec3.neg(manifold.normal);
        for (let p = 0; p < manifold.pointCount; ++p) {
            const mp = manifold.points[p];
            const tmp = mp.anchorA;
            mp.anchorA = mp.anchorB;
            mp.anchorB = tmp;
        }
    }

    const materialA = getShapeMaterial(shapeA);
    const materialB = getShapeMaterial(shapeB);

    contact.friction = world.frictionCallback(
        materialA.friction,
        materialA.userMaterialId,
        materialB.friction,
        materialB.userMaterialId,
    );
    contact.restitution = world.restitutionCallback(
        materialA.restitution,
        materialA.userMaterialId,
        materialB.restitution,
        materialB.userMaterialId,
    );

    if (materialA.rollingResistance > 0 || materialB.rollingResistance > 0) {
        const radiusA = shapeRollingRadius(shapeA);
        const radiusB = shapeRollingRadius(shapeB);
        const maxRadius = radiusA > radiusB ? radiusA : radiusB;
        const maxResist =
            materialA.rollingResistance > materialB.rollingResistance
                ? materialA.rollingResistance
                : materialB.rollingResistance;
        contact.rollingResistance = f32(maxResist * maxRadius);
    } else {
        contact.rollingResistance = 0;
    }

    const tangentVelocityA = quat.rotate(xfA.q, materialA.tangentVelocity);
    const tangentVelocityB = quat.rotate(xfB.q, materialB.tangentVelocity);
    contact.tangentVelocity = vec3.sub(tangentVelocityA, tangentVelocityB);

    if (shapeA.enableHitEvents || shapeB.enableHitEvents) {
        contact.flags |= ContactFlags.simEnableHitEvent;
    } else {
        contact.flags &= ~ContactFlags.simEnableHitEvent;
    }

    return true;
}

function shapeRollingRadius(shape: Shape): number {
    switch (shape.type) {
        case ShapeType.Sphere:
            return (shape.sphere as Sphere).radius;
        case ShapeType.Capsule:
            return (shape.capsule as Capsule).radius;
        case ShapeType.Hull:
            return f32(0.25 * (shape.hull as HullData).innerRadius);
        default:
            return 0;
    }
}

// Update a compound-vs-convex contact (the compound branch of b3UpdateContact). Resolve the contact's
// child, dispatch it as a temporary shape carrying the compound's materials, then shift the manifold
// anchors from the child origin to the compound origin. shapeA is the compound; shapeB is convex.
function updateCompoundContact(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    xfA: WorldTransform,
    shapeB: Shape,
    xfB: WorldTransform,
    isFast: boolean,
): boolean {
    const child: ChildShape = getCompoundChild(shapeA.compound as CompoundData, contact.childIndex);

    // A temporary child shape matching the convex-contact signatures. It inherits the compound's
    // materials (getShapeMaterials reads materials[0] for a convex child; a mesh child remaps per
    // triangle through child.materialIndices), with type + geometry set from the resolved child.
    const childShapeA: Shape = { ...shapeA, type: child.type };

    let touching: boolean;

    if (child.type === ShapeType.Capsule) {
        childShapeA.capsule = child.capsule;
        // Capsule/sphere children bake their transform into the geometry (child transform identity),
        // so they collide at xfA directly; the anchor offset below is then a no-op for them.
        if (shapeB.type === ShapeType.Hull) {
            touching = updateConvexContact(world, contact, shapeB, xfB, childShapeA, xfA, true);
        } else {
            touching = updateConvexContact(world, contact, childShapeA, xfA, shapeB, xfB, false);
        }
    } else if (child.type === ShapeType.Hull) {
        childShapeA.hull = child.hull;
        const xfChild = mulWorldTransforms(xfA, child.transform);
        touching = updateConvexContact(world, contact, childShapeA, xfChild, shapeB, xfB, false);
    } else if (child.type === ShapeType.Mesh) {
        childShapeA.mesh = child.mesh;
        const xfChild = mulWorldTransforms(xfA, child.transform);
        touching = computeMeshManifolds(
            world,
            contact,
            childShapeA,
            child.materialIndices,
            xfChild,
            shapeB,
            xfB,
            isFast,
        );
        if (touching && (shapeA.enableHitEvents || shapeB.enableHitEvents)) {
            contact.flags |= ContactFlags.simEnableHitEvent;
        } else {
            contact.flags &= ~ContactFlags.simEnableHitEvent;
        }
    } else {
        childShapeA.sphere = child.sphere;
        if (shapeB.type === ShapeType.Capsule || shapeB.type === ShapeType.Hull) {
            touching = updateConvexContact(world, contact, shapeB, xfB, childShapeA, xfA, true);
        } else {
            touching = updateConvexContact(world, contact, childShapeA, xfA, shapeB, xfB, false);
        }
    }

    // The anchor is relative to the child origin but oriented in world space; shift it to the compound
    // origin. Zero for capsule/sphere children (identity child transform), real for hull/mesh children.
    const offset = quat.rotate(xfA.q, child.transform.p);
    for (let m = 0; m < contact.manifoldCount; ++m) {
        const manifold = contact.manifolds[m];
        for (let p = 0; p < manifold.pointCount; ++p) {
            const mp = manifold.points[p];
            mp.anchorA = vec3.add(mp.anchorA, offset);
        }
    }

    return touching;
}

// Update a mesh / height-field / compound contact's manifold + touching status (b3UpdateContact's
// non-convex tiers). Direct convex contacts take the batched kernel dispatch (collide), not this.
function updateContact(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    localCenterA: Vec3,
    xfA: WorldTransform,
    shapeB: Shape,
    localCenterB: Vec3,
    xfB: WorldTransform,
    isFast: boolean,
): boolean {
    let touching: boolean;

    if (shapeA.type === ShapeType.Mesh || shapeA.type === ShapeType.HeightField) {
        touching = computeMeshManifolds(world, contact, shapeA, null, xfA, shapeB, xfB, isFast);
        if (touching && (shapeA.enableHitEvents || shapeB.enableHitEvents)) {
            contact.flags |= ContactFlags.simEnableHitEvent;
        } else {
            contact.flags &= ~ContactFlags.simEnableHitEvent;
        }
    } else {
        // Compound: the only other type `collide` routes here (a compound convex child still runs the
        // JS convex path via `updateCompoundContact` → `updateConvexContact`).
        touching = updateCompoundContact(world, contact, shapeA, xfA, shapeB, xfB, isFast);
    }

    if (touching) {
        const centerA = quat.rotateOut(xfA.q, localCenterA, centerAScratch);
        const centerB = quat.rotateOut(xfB.q, localCenterB, centerBScratch);
        world.manifoldStore.shiftAnchors(
            contact.contactId,
            contact.manifoldCount,
            centerA,
            centerB,
        );
        contact.flags |= ContactFlags.simTouchingFlag;
    } else {
        contact.flags &= ~ContactFlags.simTouchingFlag;
    }

    return touching;
}

// Try to reuse the existing manifold when the relative pose barely changed (recycle branch of
// b3CollideTask). Runs for nearly every contact every step, so it works entirely in module scratch;
// the per-point separation update walks the pool columns (store.recycleSeparations). @returns true
// when the contact was recycled and needs no re-collision.
function tryRecycle(
    store: ManifoldStore,
    contact: Contact,
    transformA: WorldTransform,
    transformB: WorldTransform,
    centerA: Vec3,
    centerB: Vec3,
    maxExtentA: Vec3,
    maxExtentB: Vec3,
    recycleTolerance: number,
): boolean {
    const angleA = quat.dot(transformA.q, contact.cachedRotationA);
    const angleB = quat.dot(transformB.q, contact.cachedRotationB);
    const angularDistance = minf(f32(angleA * angleA), f32(angleB * angleB));

    const xf = invMulWorldTransformsOut(transformA, transformB, recycleXf);
    const xfc = contact.cachedRelativePose;
    const maxExtent = vec3.maxOut(maxExtentA, maxExtentB, recycleExtent);

    const d = vec3.subOut(xfc.p, xf.p, recycleTmp);
    const distSquared = vec3.dot(d, d);

    if (
        angularDistance > CONTACT_RECYCLE_ANGULAR_DISTANCE &&
        distSquared < f32(recycleTolerance * recycleTolerance)
    ) {
        const distance = f32(Math.sqrt(distSquared));
        const slack = f32(recycleTolerance - distance);

        const qr = quat.invMulOut(xfc.q, xf.q, recycleQr);
        const arc = vec3.modifiedCrossOut(vec3.absOut(qr.v, recycleTmp), maxExtent, recycleTmp);
        const arcSq = f32(4.0 * vec3.lengthSq(arc));

        if (arcSq < f32(slack * slack)) {
            quat.conjugateOut(contact.cachedRotationA, recycleConj);
            quat.mulOut(transformA.q, recycleConj, recycleDq);
            mat3.fromQuatOut(recycleDq, recycleMatrixA);
            quat.conjugateOut(contact.cachedRotationB, recycleConj);
            quat.mulOut(transformB.q, recycleConj, recycleDq);
            mat3.fromQuatOut(recycleDq, recycleMatrixB);
            const dc = vec3.subOut(centerB, centerA, recycleDc);

            store.recycleSeparations(
                contact.contactId,
                contact.manifoldCount,
                recycleMatrixA,
                recycleMatrixB,
                dc,
            );
            return true;
        }
    }

    return false;
}

// Park a contact in the awake set's non-touching list (b3AddNonTouchingContact).
function addNonTouchingContact(world: WorldState, contact: Contact): void {
    const set = world.solverSets[SetType.Awake];
    contact.colorIndex = NULL_INDEX;
    contact.localIndex = set.contactIndices.length;
    set.contactIndices.push(contact.contactId);
}

// Remove a contact from a set's non-touching list (b3RemoveNonTouchingContact).
function removeNonTouchingContact(world: WorldState, setIndex: number, localIndex: number): void {
    const set = world.solverSets[setIndex];
    const movedIndex = swapRemove(set.contactIndices, localIndex);
    if (movedIndex !== NULL_INDEX) {
        const movedContactIndex = set.contactIndices[localIndex];
        world.contacts[movedContactIndex].localIndex = localIndex;
    }
}

// A direct convex contact deferred to the batched kernel dispatch. Carries the shapes + transforms the
// finish pass needs for material mixing, tangent velocity, and the center-of-mass anchor shift; the
// manifold itself is written column-resident by the kernel over the shared columns.
type ConvexJob = {
    contact: Contact;
    shapeA: Shape;
    shapeB: Shape;
    xfA: WorldTransform;
    xfB: WorldTransform;
    localCenterA: Vec3;
    localCenterB: Vec3;
    wasTouching: boolean;
};

// Job records recycled across steps (the live prefix is jobCount); fields are overwritten on collect,
// so a record may hold references from the previous step until then — bounded, world-lived objects.
const jobPool: ConvexJob[] = [];
let jobCount = 0;

// Write a world transform (p3 + q4) into a dispatch record at `o` (columns.ts D_XF_*).
function writeDispatchXf(f: Float32Array, o: number, xf: WorldTransform): void {
    f[o] = xf.p.x;
    f[o + 1] = xf.p.y;
    f[o + 2] = xf.p.z;
    f[o + 3] = xf.q.v.x;
    f[o + 4] = xf.q.v.y;
    f[o + 5] = xf.q.v.z;
    f[o + 6] = xf.q.s;
}

// Write a convex shape's geom slots into a dispatch record at `o`: a hull encodes its uploaded geoIndex
// (u32), a sphere its center + radius, a capsule its two centers + radius (columns.ts D_GEOM_*).
function writeDispatchGeom(f: Float32Array, u: Uint32Array, o: number, shape: Shape): void {
    if (shape.type === ShapeType.Hull) {
        u[o] = (shape.hull as HullData).geoIndex;
    } else if (shape.type === ShapeType.Sphere) {
        const s = shape.sphere as Sphere;
        f[o] = s.center.x;
        f[o + 1] = s.center.y;
        f[o + 2] = s.center.z;
        f[o + 3] = s.radius;
    } else {
        const c = shape.capsule as Capsule;
        f[o] = c.center1.x;
        f[o + 1] = c.center1.y;
        f[o + 2] = c.center1.z;
        f[o + 3] = c.center2.x;
        f[o + 4] = c.center2.y;
        f[o + 5] = c.center2.z;
        f[o + 6] = c.radius;
    }
}

// Ensure a direct convex contact's persistent manifold block exists for the kernel to write into, and
// defer the contact to the batched dispatch. A non-touching contact (manifoldCount 0) allocates a fresh
// block (recycling a freed one) and zeros its warm-start header + point count so the kernel reads an
// empty old manifold (matching b3ComputeConvexManifold's fresh-manifold init); a still-touching contact
// reuses its block in place, last step's manifold being the warm-start source. The manifold + material
// are computed later — the kernel writes the manifold, the finish pass the material.
function collectConvex(
    world: WorldState,
    contact: Contact,
    shapeA: Shape,
    xfA: WorldTransform,
    localCenterA: Vec3,
    shapeB: Shape,
    xfB: WorldTransform,
    localCenterB: Vec3,
    wasTouching: boolean,
): void {
    if (contact.manifoldCount === 0) {
        contact.manifolds = world.manifoldStore.alloc(contact.contactId, 1);
        contact.manifoldCount = 1;
        const manifold = contact.manifolds[0];
        manifold.frictionImpulse = zero;
        manifold.rollingImpulse = zero;
        manifold.twistImpulse = 0;
        manifold.pointCount = 0;
    }
    const job = jobPool[jobCount];
    if (job === undefined) {
        jobPool.push({
            contact,
            shapeA,
            shapeB,
            xfA,
            xfB,
            localCenterA,
            localCenterB,
            wasTouching,
        });
    } else {
        job.contact = contact;
        job.shapeA = shapeA;
        job.shapeB = shapeB;
        job.xfA = xfA;
        job.xfB = xfB;
        job.localCenterA = localCenterA;
        job.localCenterB = localCenterB;
        job.wasTouching = wasTouching;
    }
    ++jobCount;
}

// Finish a dispatched direct convex contact after the kernel wrote (or cleared) its manifold: apply the
// material/tangent/hit updates (b3UpdateConvexContact's tail — no flip, a direct convex contact is never
// flipped) and the center-of-mass anchor shift (b3UpdateContact's tail), record the touching transition,
// and cache separations for the next recycle test. A non-touching contact frees its speculative block.
function finishConvex(
    world: WorldState,
    job: ConvexJob,
    touching: boolean,
    stateChanges: number[],
): void {
    const contact = job.contact;
    if (touching === false) {
        contact.manifolds = NO_MANIFOLDS;
        contact.manifoldCount = 0;
        world.manifoldStore.clear(contact.contactId);
        contact.flags &= ~ContactFlags.simTouchingFlag;
        if (job.wasTouching) {
            contact.flags |= ContactFlags.simStoppedTouching;
            stateChanges.push(contact.contactId);
        }
        return;
    }

    const shapeA = job.shapeA;
    const shapeB = job.shapeB;
    const materialA = getShapeMaterial(shapeA);
    const materialB = getShapeMaterial(shapeB);

    contact.friction = world.frictionCallback(
        materialA.friction,
        materialA.userMaterialId,
        materialB.friction,
        materialB.userMaterialId,
    );
    contact.restitution = world.restitutionCallback(
        materialA.restitution,
        materialA.userMaterialId,
        materialB.restitution,
        materialB.userMaterialId,
    );

    if (materialA.rollingResistance > 0 || materialB.rollingResistance > 0) {
        const radiusA = shapeRollingRadius(shapeA);
        const radiusB = shapeRollingRadius(shapeB);
        const maxRadius = radiusA > radiusB ? radiusA : radiusB;
        const maxResist =
            materialA.rollingResistance > materialB.rollingResistance
                ? materialA.rollingResistance
                : materialB.rollingResistance;
        contact.rollingResistance = f32(maxResist * maxRadius);
    } else {
        contact.rollingResistance = 0;
    }

    const tangentVelocityA = quat.rotateOut(job.xfA.q, materialA.tangentVelocity, tangentAScratch);
    const tangentVelocityB = quat.rotateOut(job.xfB.q, materialB.tangentVelocity, tangentBScratch);
    vec3.subOut(tangentVelocityA, tangentVelocityB, contact.tangentVelocity);

    if (shapeA.enableHitEvents || shapeB.enableHitEvents) {
        contact.flags |= ContactFlags.simEnableHitEvent;
    } else {
        contact.flags &= ~ContactFlags.simEnableHitEvent;
    }

    // b3UpdateContact tail: shift anchors from body origin to center of mass.
    const centerA = quat.rotateOut(job.xfA.q, job.localCenterA, centerAScratch);
    const centerB = quat.rotateOut(job.xfB.q, job.localCenterB, centerBScratch);
    world.manifoldStore.shiftAnchors(contact.contactId, contact.manifoldCount, centerA, centerB);
    contact.flags |= ContactFlags.simTouchingFlag;

    if (job.wasTouching === false) {
        contact.flags |= ContactFlags.simStartedTouching;
        stateChanges.push(contact.contactId);
    }

    world.manifoldStore.rebaseSeparations(contact.contactId, contact.manifoldCount);
}

// Run every collected direct convex contact through the kernel narrowphase in one batched call, then
// finish each. A mid-collect pool grow (mesh or the speculative convex alloc) shifted the geometry region
// the dispatch reads its hulls from, so re-upload it (fresh geoIndex + GEO_END past the grown pool) before
// reserving the dispatch column at the solver base. The reservations grow wasm memory, detaching the store
// views the finish pass reads through, so re-derive them after the last grow.
function dispatchConvexJobs(world: WorldState, stateChanges: number[]): void {
    if (world.manifoldStore.grew) {
        rebuildGeometry(world);
        world.manifoldStore.grew = false;
        world.bodyStore.refreshViews();
    }

    const k = kernel();
    k.reserveDispatch(jobCount);
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();

    const buf = k.memory.buffer;
    const dispF = new Float32Array(buf, k.dispatchPtr(), jobCount * DISPATCH_STRIDE);
    const dispU = new Uint32Array(buf, k.dispatchPtr(), jobCount * DISPATCH_STRIDE);
    for (let i = 0; i < jobCount; ++i) {
        const job = jobPool[i];
        const r = i * DISPATCH_STRIDE;
        dispU[r + D_CONTACT] = job.contact.contactId;
        dispU[r + D_TYPE_A] = job.shapeA.type;
        dispU[r + D_TYPE_B] = job.shapeB.type;
        writeDispatchXf(dispF, r + D_XF_A, job.xfA);
        writeDispatchXf(dispF, r + D_XF_B, job.xfB);
        writeDispatchGeom(dispF, dispU, r + D_GEOM_A, job.shapeA);
        writeDispatchGeom(dispF, dispU, r + D_GEOM_B, job.shapeB);
    }

    // Across the pool when there is one: each record's narrowphase reads its own dispatch record and
    // writes only its own contact's manifold + cache slots, so the blocks are write-disjoint. The reserve
    // above is the last grow before the fork.
    runPar(ParKind.Convex, jobCount, 0, 0, () => k.dispatchConvex(jobCount));

    const out = new Uint32Array(buf, k.dispatchOutPtr(), jobCount);
    for (let i = 0; i < jobCount; ++i) {
        finishConvex(world, jobPool[i], out[i] !== 0, stateChanges);
    }
}

// Run every gathered dynamic-dynamic convex contact through the batched kernel recycle pass (overlap +
// recycle-gate + separation update + pose-cache over the resident body / fat-AABB / manifold columns),
// then act on each result. A recycled contact (0) is finished in-kernel — its manifold separations were
// updated and it skips the narrowphase. A needs-narrowphase contact (1) had its pose cached in-kernel; TS
// marks it relative-transform-valid and re-fetches the bodies' sims to collect it into the convex dispatch.
// A disjoint contact (2) is torn down like the TS fat-AABB miss. The reserve grows wasm memory (detaching
// the store views the finish reads through), so re-derive them after it, and after a mid-loop manifold-pool
// grow (the same detach guard the collect loop uses).
function dispatchRecycleJobs(
    world: WorldState,
    count: number,
    recycleDistance: number,
    recycleDistanceNonTouching: number,
    stateChanges: number[],
): void {
    // A mid-collect pool grow (mesh / static-convex alloc) shifted the geometry region; re-upload it before
    // the reserve places the recycle column at the solver base past it, mirroring dispatchConvexJobs.
    if (world.manifoldStore.grew) {
        rebuildGeometry(world);
        world.manifoldStore.grew = false;
        world.bodyStore.refreshViews();
    }

    const k = kernel();
    k.reserveRecycle(count);
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();

    const recycle = world.awakeRecycleContacts;
    const buf = k.memory.buffer;
    const inU = new Uint32Array(buf, k.recyclePtr(), count * RECYCLE_STRIDE);
    for (let i = 0; i < count; ++i) {
        const contactId = recycle[i];
        const contact = world.contacts[contactId];
        const flags = contact.flags;
        const r = i * RECYCLE_STRIDE;
        inU[r + R_CONTACT] = contactId;
        inU[r + R_LOCAL_A] = contact.bodySimIndexA;
        inU[r + R_LOCAL_B] = contact.bodySimIndexB;
        inU[r + R_SHAPE_A] = contact.shapeIdA;
        inU[r + R_SHAPE_B] = contact.shapeIdB;
        let bits = 0;
        if (
            recycleDistance > 0 &&
            (flags & ContactFlags.relativeTransformValid) !== 0 &&
            (flags & ContactFlags.contactRecycleFlag) !== 0
        ) {
            bits |= R_ELIGIBLE;
        }
        if ((flags & ContactFlags.simTouchingFlag) !== 0) {
            bits |= R_WAS_TOUCHING;
        }
        inU[r + R_BITS] = bits;
    }

    // Across the pool when there is one: the body / fat-AABB columns are read-only here and every write
    // lands in the record's own contact's slots, so the blocks are write-disjoint. The reserve above is
    // the last grow before the fork.
    runPar(ParKind.Recycle, count, recycleDistance, recycleDistanceNonTouching, () =>
        k.dispatchRecycle(count, recycleDistance, recycleDistanceNonTouching),
    );

    // Snapshot every result out of the kernel's output column *before* processing. The needs-narrowphase
    // branch below calls `collectConvex`, whose manifold `alloc` can grow/move the pool — detaching the
    // typed-array view (or, when the reserve already grew memory, overwriting the output bytes in place
    // with pool data). Reading a clobbered result as 0 would silently drop a contact's narrowphase, so
    // copy the whole column into a plain array while it is still valid.
    const out = new Uint32Array(buf, k.recycleOutPtr(), count);
    for (let i = 0; i < count; ++i) recycleResults[i] = out[i];

    for (let i = 0; i < count; ++i) {
        const contactId = recycle[i];
        const contact = world.contacts[contactId];
        const result = recycleResults[i];
        if (result === 0) {
            // Recycled — the manifold separations were updated in-kernel; nothing left to do.
            continue;
        }
        if (result === 2) {
            // Disjoint — the fat AABBs no longer overlap (b3Collide's disjoint branch).
            contact.flags |= ContactFlags.simDisjoint;
            contact.flags &= ~ContactFlags.simTouchingFlag;
            stateChanges.push(contactId);
            continue;
        }
        // Needs full narrowphase. The kernel cached this step's pose; mark it valid and collect the
        // contact for the convex dispatch. Re-derive the views if a prior collect's alloc grew the pool.
        contact.flags |= ContactFlags.relativeTransformValid;
        if (world.bodyStore.stale) world.bodyStore.refreshViews();
        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        const bodySimA = getBodySim(world, world.bodies[shapeA.bodyId]);
        const bodySimB = getBodySim(world, world.bodies[shapeB.bodyId]);
        collectConvex(
            world,
            contact,
            shapeA,
            bodySimA.transform,
            bodySimA.localCenter,
            shapeB,
            bodySimB.transform,
            bodySimB.localCenter,
            (contact.flags & ContactFlags.simTouchingFlag) !== 0,
        );
    }
}

/** Narrow-phase collision + touching-state transitions for all awake contacts (b3Collide). Direct convex
 * contacts (shapeA sphere/capsule/hull) are batched into the wasm kernel narrowphase; mesh, height-field,
 * and compound contacts run the TS narrowphase inline (the disjoint shape-pair partition). */
export function collide(context: StepContext): void {
    const world = context.world;

    // Re-derive the manifold store's column views: the geometry re-upload (step.ts) may have grown wasm
    // memory since the store last refreshed, detaching them, and the narrowphase reads/writes the pool.
    world.manifoldStore.refreshViews();

    // The awake contacts are pre-partitioned into two incremental lists (world.awake*Contacts), maintained
    // on contact create/destroy + body wake/sleep/transfer (contact.ts, solverset.ts) instead of gathered
    // per step. `awakeRecycleContacts` (dynamic-dynamic direct-convex, both bodies awake) takes the batched
    // kernel recycle pass; `awakeOtherContacts` (static-involved / mesh / height-field / compound /
    // sleeping-partner) runs the TS per-contact walk below. A static/sleeping-partner convex contact's
    // narrowphase still batches to the kernel (collectConvex); only its recycle fast path stays TS.
    const recycleCount = world.awakeRecycleContacts.length;
    const otherContacts = world.awakeOtherContacts;

    if (recycleCount === 0 && otherContacts.length === 0) {
        return;
    }

    const recycleDistance = world.contactRecycleDistance;
    const recycleDistanceNonTouching = minf(recycleDistance, SPECULATIVE_DISTANCE);

    // Contacts whose touching state changed this step, processed in ascending id order below.
    const stateChanges = stateChangeIds;
    stateChanges.length = 0;
    // Direct convex contacts deferred to the batched kernel dispatch after the collect loop.
    jobCount = 0;

    for (let i = 0; i < otherContacts.length; ++i) {
        const contactId = otherContacts[i];
        const contact = world.contacts[contactId];

        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        const bodyA = world.bodies[shapeA.bodyId];
        const bodyB = world.bodies[shapeB.bodyId];

        // Recycle eligibility's per-contact-constant half — a would-be-recycle contact currently off the
        // kernel path (a partner is still asleep) keeps the *same* directory pose cache (readRecyclePose/
        // writeRecyclePose) as the kernel path, so recycling stays consistent across the sleep/wake
        // transition. The mesh/height-field/compound/static contacts (recycleStable false) use the JS
        // `cachedRotation*` fields instead. Both-bodies-awake convex contacts never reach here — they live
        // in awakeRecycleContacts (the kernel recycle pass).
        const stableConvex = contact.recycleStable;

        // Do the fat AABBs still overlap?
        if (aabb.overlaps(shapeA.fatAABB, shapeB.fatAABB) === false) {
            contact.flags |= ContactFlags.simDisjoint;
            contact.flags &= ~ContactFlags.simTouchingFlag;
            stateChanges.push(contactId);
            continue;
        }

        const isStaticA = bodyA.type === BodyType.Static;
        const isStaticB = bodyB.type === BodyType.Static;
        const wasTouching = (contact.flags & ContactFlags.simTouchingFlag) !== 0;
        const isMeshContact = (contact.flags & ContactFlags.simMeshContact) !== 0;

        // A prior contact's manifold `alloc` may have grown the pool (memory.grow), detaching the
        // resident body-store column views the awake `BodySim` reads below go through — leaving them
        // pointing at a detached buffer (length 0 → NaN reads). Re-derive them before reading. Guarded on
        // the grow (`stale`, bodycolumns.ts) so it fires only on an actual one (a few times per step),
        // not per contact (churn). The plain-object sims this replaced were immune; the columns are not.
        if (world.bodyStore.stale) world.bodyStore.refreshViews();

        const bodySimA = getBodySim(world, bodyA);
        const bodySimB = getBodySim(world, bodyB);
        const transformA = bodySimA.transform;
        const transformB = bodySimB.transform;
        const isFast =
            (bodySimA.flags & BodyFlags.isFast) !== 0 || (bodySimB.flags & BodyFlags.isFast) !== 0;

        const recycleTolerance = wasTouching ? recycleDistance : recycleDistanceNonTouching;

        // Contact recycling: reuse the manifold when the relative pose barely moved. A fast mesh
        // contact skips recycling — the hull can rotate around a triangle edge and tunnel.
        if (
            (isFast === false || isMeshContact === false) &&
            recycleDistance > 0 &&
            (contact.flags & ContactFlags.relativeTransformValid) !== 0 &&
            (contact.flags & ContactFlags.contactRecycleFlag) !== 0
        ) {
            // A would-be-kernel contact off the kernel path (a partner is asleep) reads its cached pose
            // from the directory (the kernel wrote it last time it ran the kernel path), not the stale
            // JS fields, so the recycle test is bit-identical across the transition.
            if (stableConvex) {
                world.manifoldStore.readRecyclePose(
                    contactId,
                    contact.cachedRotationA,
                    contact.cachedRotationB,
                    contact.cachedRelativePose,
                );
            }
            const maxExtentA = isStaticA ? zero : bodySimA.maxExtent;
            const maxExtentB = isStaticB ? zero : bodySimB.maxExtent;
            if (
                tryRecycle(
                    world.manifoldStore,
                    contact,
                    transformA,
                    transformB,
                    bodySimA.center,
                    bodySimB.center,
                    maxExtentA,
                    maxExtentB,
                    recycleTolerance,
                )
            ) {
                continue;
            }
        }

        // Cache the relative pose for the next step's recycling test. The rotations are copied out
        // of the live body sims component-wise (detaching them); the cached objects are contact-owned
        // and mutated in place, never aliased to the sims.
        quat.copy(transformA.q, contact.cachedRotationA);
        quat.copy(transformB.q, contact.cachedRotationB);
        invMulWorldTransformsOut(transformA, transformB, contact.cachedRelativePose);
        contact.flags |= ContactFlags.relativeTransformValid;
        // Mirror the pose into the directory for a would-be-kernel contact, so the next step's kernel
        // pass (if the sleeping partner has woken by then) reads a current cache.
        if (stableConvex) {
            world.manifoldStore.writeRecyclePose(
                contactId,
                contact.cachedRotationA,
                contact.cachedRotationB,
                contact.cachedRelativePose,
            );
        }

        // The partition: mesh / height-field / compound run the TS narrowphase inline; direct convex
        // (sphere / capsule / hull) defers to the batched kernel dispatch below.
        if (
            shapeA.type === ShapeType.Mesh ||
            shapeA.type === ShapeType.HeightField ||
            shapeA.type === ShapeType.Compound
        ) {
            const touching = updateContact(
                world,
                contact,
                shapeA,
                bodySimA.localCenter,
                transformA,
                shapeB,
                bodySimB.localCenter,
                transformB,
                isFast,
            );

            if (touching === true && wasTouching === false) {
                contact.flags |= ContactFlags.simStartedTouching;
                stateChanges.push(contactId);
            } else if (touching === false && wasTouching === true) {
                contact.flags |= ContactFlags.simStoppedTouching;
                stateChanges.push(contactId);
            }

            // Cache separations for the next recycle test.
            world.manifoldStore.rebaseSeparations(contactId, contact.manifoldCount);
        } else {
            collectConvex(
                world,
                contact,
                shapeA,
                transformA,
                bodySimA.localCenter,
                shapeB,
                transformB,
                bodySimB.localCenter,
                wasTouching,
            );
        }
    }

    // Run the batched kernel recycle pass over the gathered dynamic-dynamic convex contacts. Recycled
    // contacts are done in-kernel; the needs-narrowphase minority feeds the convex dispatch below (via
    // collectConvex), and disjoint contacts push a state change. Runs before the dispatch so its
    // needs-narrowphase jobs join the same batched narrowphase call.
    if (recycleCount > 0) {
        dispatchRecycleJobs(
            world,
            recycleCount,
            recycleDistance,
            recycleDistanceNonTouching,
            stateChanges,
        );
    }

    if (jobCount > 0) {
        dispatchConvexJobs(world, stateChanges);
    }

    // Serial touching-state transitions, in ascending contact id order (matches C's bit-set walk).
    stateChanges.sort((a, b) => a - b);
    const endEventArrayIndex = world.endEventArrayIndex;
    const worldId = world.worldId;

    for (const contactId of stateChanges) {
        const contact = world.contacts[contactId];
        const shapeA = world.shapes[contact.shapeIdA];
        const shapeB = world.shapes[contact.shapeIdB];
        const flags = contact.flags;

        if (flags & ContactFlags.simDisjoint) {
            destroyContact(world, contact, false);
        } else if (flags & ContactFlags.simStartedTouching) {
            if (flags & ContactFlags.contactEnableContactEvents) {
                world.contactBeginEvents.push({
                    shapeIdA: {
                        index1: shapeA.id + 1,
                        world0: worldId,
                        generation: shapeA.generation,
                    },
                    shapeIdB: {
                        index1: shapeB.id + 1,
                        world0: worldId,
                        generation: shapeB.generation,
                    },
                    contactId: {
                        index1: contact.contactId + 1,
                        world0: worldId,
                        generation: contact.generation,
                    },
                });
            }

            // Link first because this wakes colliding bodies and moves body sims into place.
            contact.flags &= ~ContactFlags.simStartedTouching;
            contact.flags |= ContactFlags.contactTouchingFlag;
            linkContact(world, contact);

            const oldLocalIndex = contact.localIndex;
            addContactToGraph(world, contact);
            removeNonTouchingContact(world, SetType.Awake, oldLocalIndex);
        } else if (flags & ContactFlags.simStoppedTouching) {
            contact.flags &= ~ContactFlags.simStoppedTouching;
            contact.flags &= ~ContactFlags.contactTouchingFlag;

            if (contact.flags & ContactFlags.contactEnableContactEvents) {
                world.contactEndEvents[endEventArrayIndex].push({
                    shapeIdA: {
                        index1: shapeA.id + 1,
                        world0: worldId,
                        generation: shapeA.generation,
                    },
                    shapeIdB: {
                        index1: shapeB.id + 1,
                        world0: worldId,
                        generation: shapeB.generation,
                    },
                    contactId: {
                        index1: contact.contactId + 1,
                        world0: worldId,
                        generation: contact.generation,
                    },
                });
            }

            const colorIndex = contact.colorIndex;
            const localIndex = contact.localIndex;

            unlinkContact(world, contact);
            addNonTouchingContact(world, contact);
            removeContactFromGraph(
                world,
                contact.edges[0].bodyId,
                contact.edges[1].bodyId,
                colorIndex,
                localIndex,
                (contact.flags & ContactFlags.simMeshContact) !== 0,
            );
        }
    }
}
