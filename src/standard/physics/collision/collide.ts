// Box3D's b3Collide: the kernel computes contact geometry; touching transitions, material callbacks,
// events, island linking and constraint-graph moves remain on the calling thread.
import { NULL_INDEX, swapRemove } from "../common/array";
import { SetType, SPECULATIVE_DISTANCE } from "../common/constants";
import {
    f32,
    minf,
    mulWorldTransforms,
    quat,
    type Vec3,
    vec3,
    type WorldTransform,
} from "../common/math";
import { BodyType, ShapeType } from "../common/types";
import {
    D_CACHE_VALID,
    D_CHILD,
    D_CONTACT,
    D_FAST,
    D_GEOM_A,
    D_GEOM_B,
    D_LOWER,
    D_MESH_SLOT,
    D_OLD_COUNT,
    D_TYPE_A,
    D_TYPE_B,
    D_UPPER,
    D_XF_A,
    D_XF_B,
    DISPATCH_STRIDE,
    R_BITS,
    R_CONTACT,
    R_COUNT,
    R_ELIGIBLE,
    R_FALLBACK_A,
    R_FALLBACK_B,
    R_LOCAL_A,
    R_LOCAL_B,
    R_SHAPE_A,
    R_SHAPE_B,
    R_WAS_TOUCHING,
    RECYCLE_STRIDE,
} from "../kernel/columns";
import { rebuildGeometry } from "../kernel/geocolumns";
import { kernel, ParKind, runPar, threads } from "../kernel/kernel";
import { getCompoundChild } from "../shapes/compound";
import {
    getShapeMaterial,
    getShapeMaterialCount,
    getShapeMaterials,
    type Shape,
} from "../shapes/shape";
import type { StepContext } from "../solver/contactsolver";
import { addContactToGraph, removeContactFromGraph } from "../solver/graph";
import { BodyFlags, getBodySim } from "../world/body";
import { linkContact, unlinkContact } from "../world/island";
import type { WorldState } from "../world/world";
import { type Contact, ContactFlags, destroyContact, type Manifold } from "./contact";
import { MANIFOLD_STRIDE } from "./manifoldstore";

const stateChanges: number[] = [];
const results: number[] = [];
const zero: Vec3 = { x: 0, y: 0, z: 0 };
const NO_MANIFOLDS: Manifold[] = [];
const centerA = vec3.zero();
const centerB = vec3.zero();
const tangentA = vec3.zero();
const tangentB = vec3.zero();

type ContactJob = {
    contact: Contact;
    shapeA: Shape;
    shapeB: Shape;
    xfA: WorldTransform;
    xfB: WorldTransform;
    localCenterA: Vec3;
    localCenterB: Vec3;
    wasTouching: boolean;
    isFast: boolean;
    meshSlot: number;
    result: Uint32Array | null;
    materials: Uint32Array | null;
};
const jobs: ContactJob[] = [];
let jobCount = 0;

function writeXf(f: Float32Array, o: number, xf: WorldTransform): void {
    f[o] = xf.p.x;
    f[o + 1] = xf.p.y;
    f[o + 2] = xf.p.z;
    f[o + 3] = xf.q.v.x;
    f[o + 4] = xf.q.v.y;
    f[o + 5] = xf.q.v.z;
    f[o + 6] = xf.q.s;
}
function writeVec(f: Float32Array, o: number, v: Vec3): void {
    f[o] = v.x;
    f[o + 1] = v.y;
    f[o + 2] = v.z;
}
function writeGeom(
    world: WorldState,
    f: Float32Array,
    u: Uint32Array,
    o: number,
    shape: Shape,
): void {
    switch (shape.type) {
        case ShapeType.Hull:
            u[o] = shape.hull!.geoIndex;
            break;
        case ShapeType.Sphere:
            writeVec(f, o, shape.sphere!.center);
            f[o + 3] = shape.sphere!.radius;
            break;
        case ShapeType.Capsule:
            writeVec(f, o, shape.capsule!.center1);
            writeVec(f, o + 3, shape.capsule!.center2);
            f[o + 6] = shape.capsule!.radius;
            break;
        case ShapeType.Mesh:
            u[o] = world.meshDatabase.get(shape.mesh!.data)!.geoIndex;
            writeVec(f, o + 1, shape.mesh!.scale);
            break;
        case ShapeType.HeightField:
            u[o] = world.heightFieldDatabase.get(shape.heightField!)!.geoIndex;
            break;
        case ShapeType.Compound:
            u[o] = world.compoundDatabase.get(shape.compound!)!.geoIndex;
            break;
    }
}
function rollingRadius(shape: Shape): number {
    switch (shape.type) {
        case ShapeType.Sphere:
            return shape.sphere!.radius;
        case ShapeType.Capsule:
            return shape.capsule!.radius;
        case ShapeType.Hull:
            return f32(0.25 * shape.hull!.innerRadius);
        default:
            return 0;
    }
}
function collect(world: WorldState, contact: Contact): void {
    if (world.bodyStore.stale) world.bodyStore.refreshViews();
    const shapeA = world.shapes[contact.shapeIdA];
    const shapeB = world.shapes[contact.shapeIdB];
    const simA = getBodySim(world, world.bodies[shapeA.bodyId]);
    const simB = getBodySim(world, world.bodies[shapeB.bodyId]);
    const mesh = (contact.flags & ContactFlags.simMeshContact) !== 0;
    if (!mesh && contact.manifoldCount === 0) {
        contact.manifolds = world.manifoldStore.alloc(contact.contactId, 1);
        contact.manifoldCount = 1;
        const m = contact.manifolds[0];
        m.frictionImpulse = zero;
        m.rollingImpulse = zero;
        m.twistImpulse = 0;
        m.pointCount = 0;
    }
    if (world.bodyStore.stale) world.bodyStore.refreshViews();
    let job = jobs[jobCount];
    if (job === undefined) {
        job = {
            contact,
            shapeA,
            shapeB,
            xfA: simA.transform,
            xfB: simB.transform,
            localCenterA: simA.localCenter,
            localCenterB: simB.localCenter,
            wasTouching: false,
            isFast: false,
            meshSlot: -1,
            result: null,
            materials: null,
        };
        jobs.push(job);
    }
    job.contact = contact;
    job.shapeA = shapeA;
    job.shapeB = shapeB;
    job.xfA = simA.transform;
    job.xfB = simB.transform;
    job.localCenterA = simA.localCenter;
    job.localCenterB = simB.localCenter;
    job.wasTouching = (contact.flags & ContactFlags.simTouchingFlag) !== 0;
    job.isFast = ((simA.flags | simB.flags) & BodyFlags.isFast) !== 0;
    job.meshSlot = mesh ? 0 : -1;
    ++jobCount;
}
function finishMeshMaterial(
    world: WorldState,
    job: ContactJob,
    shapeA: Shape,
    xfA: WorldTransform,
    materialMap: number[] | null,
): void {
    const contact = job.contact;
    const materialsA = getShapeMaterials(world.ecsState, shapeA);
    const materialB = getShapeMaterial(world.ecsState, job.shapeB);
    vec3.copy(zero, tangentA);
    const materialCount = getShapeMaterialCount(world.ecsState, shapeA);
    if (materialCount > 0) {
        let friction = 0,
            restitution = 0,
            samples = 0;
        for (let i = 0; i < contact.manifoldCount; ++i) {
            const m = contact.manifolds[i];
            for (let j = 0; j < m.pointCount; ++j) {
                let index = job.materials![i * 4 + j];
                if (materialMap !== null) index = materialMap[index];
                index = Math.max(0, Math.min(index, materialCount - 1));
                const material = materialsA[index];
                friction = f32(
                    friction +
                        world.frictionCallback(
                            material.friction,
                            material.userMaterialId,
                            materialB.friction,
                            materialB.userMaterialId,
                        ),
                );
                restitution = f32(
                    restitution +
                        world.restitutionCallback(
                            material.restitution,
                            material.userMaterialId,
                            materialB.restitution,
                            materialB.userMaterialId,
                        ),
                );
                vec3.addOut(tangentA, material.tangentVelocity, tangentA);
                samples = f32(samples + 1);
            }
        }
        if (samples > 0) {
            const inv = f32(1 / samples);
            contact.friction = f32(inv * friction);
            contact.restitution = f32(inv * restitution);
            vec3.scaleOut(inv, tangentA, tangentA);
        }
    } else {
        const material = materialsA[0];
        contact.friction = world.frictionCallback(
            material.friction,
            material.userMaterialId,
            materialB.friction,
            materialB.userMaterialId,
        );
        contact.restitution = world.restitutionCallback(
            material.restitution,
            material.userMaterialId,
            materialB.restitution,
            materialB.userMaterialId,
        );
        vec3.copy(material.tangentVelocity, tangentA);
    }
    const radius =
        job.shapeB.type === ShapeType.Hull
            ? job.shapeB.hull!.innerRadius
            : rollingRadius(job.shapeB);
    contact.rollingResistance = f32(materialB.rollingResistance * radius);
    quat.rotateOut(xfA.q, tangentA, tangentA);
    quat.rotateOut(job.xfB.q, materialB.tangentVelocity, tangentB);
    vec3.subOut(tangentA, tangentB, contact.tangentVelocity);
}
function finish(world: WorldState, job: ContactJob, count: number): void {
    const contact = job.contact;
    if (count === 0) {
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
    if (job.meshSlot !== -1) {
        contact.manifolds = world.manifoldStore.importManifolds(
            contact.contactId,
            count,
            job.result!,
        );
        contact.manifoldCount = count;
    }
    if (world.bodyStore.stale) world.bodyStore.refreshViews();
    let shapeA = job.shapeA,
        shapeB = job.shapeB;
    let xfA = job.xfA,
        xfB = job.xfB;
    let materialMap: number[] | null = null;
    if (shapeA.type === ShapeType.Compound) {
        const child = getCompoundChild(shapeA.compound!, contact.childIndex);
        shapeA = {
            ...shapeA,
            type: child.type,
            sphere: child.sphere,
            capsule: child.capsule,
            hull: child.hull,
            mesh: child.mesh,
        };
        materialMap = child.materialIndices;
        if (child.type === ShapeType.Hull || child.type === ShapeType.Mesh)
            xfA = mulWorldTransforms(job.xfA, child.transform);
        if (
            (child.type === ShapeType.Sphere && shapeB.type !== ShapeType.Sphere) ||
            (child.type === ShapeType.Capsule && shapeB.type === ShapeType.Hull)
        ) {
            const shape = shapeA;
            shapeA = shapeB;
            shapeB = shape;
            const xf = xfA;
            xfA = xfB;
            xfB = xf;
        }
    }
    if (job.meshSlot !== -1) {
        finishMeshMaterial(world, job, shapeA, xfA, materialMap);
    } else {
        const a = getShapeMaterial(world.ecsState, shapeA),
            b = getShapeMaterial(world.ecsState, shapeB);
        contact.friction = world.frictionCallback(
            a.friction,
            a.userMaterialId,
            b.friction,
            b.userMaterialId,
        );
        contact.restitution = world.restitutionCallback(
            a.restitution,
            a.userMaterialId,
            b.restitution,
            b.userMaterialId,
        );
        contact.rollingResistance =
            a.rollingResistance > 0 || b.rollingResistance > 0
                ? f32(
                      Math.max(a.rollingResistance, b.rollingResistance) *
                          Math.max(rollingRadius(shapeA), rollingRadius(shapeB)),
                  )
                : 0;
        quat.rotateOut(xfA.q, a.tangentVelocity, tangentA);
        quat.rotateOut(xfB.q, b.tangentVelocity, tangentB);
        vec3.subOut(tangentA, tangentB, contact.tangentVelocity);
    }
    if (job.shapeA.enableHitEvents || job.shapeB.enableHitEvents)
        contact.flags |= ContactFlags.simEnableHitEvent;
    else contact.flags &= ~ContactFlags.simEnableHitEvent;
    quat.rotateOut(job.xfA.q, job.localCenterA, centerA);
    quat.rotateOut(job.xfB.q, job.localCenterB, centerB);
    world.manifoldStore.shiftAnchors(contact.contactId, contact.manifoldCount, centerA, centerB);
    contact.flags |= ContactFlags.simTouchingFlag;
    if (!job.wasTouching) {
        contact.flags |= ContactFlags.simStartedTouching;
        stateChanges.push(contact.contactId);
    }
    world.manifoldStore.rebaseSeparations(contact.contactId, contact.manifoldCount);
}
function refreshGeometry(world: WorldState): void {
    if (world.manifoldStore.grew) {
        rebuildGeometry(world);
        world.manifoldStore.grew = false;
    }
}
function dispatch(world: WorldState): void {
    refreshGeometry(world);
    const k = kernel(world.ecsState);
    let meshCount = 0;
    for (let i = 0; i < jobCount; ++i) if (jobs[i].meshSlot !== -1) jobs[i].meshSlot = meshCount++;
    k.reserveDispatch(jobCount, meshCount, threads(world.ecsState));
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    const buf = k.memory.buffer;
    const f = new Float32Array(buf, k.dispatchPtr(), jobCount * DISPATCH_STRIDE);
    const u = new Uint32Array(buf, k.dispatchPtr(), jobCount * DISPATCH_STRIDE);
    const cacheWords = k.meshCacheBytes() / 4;
    for (let i = 0; i < jobCount; ++i) {
        const job = jobs[i],
            r = i * DISPATCH_STRIDE;
        u[r + D_CONTACT] = job.contact.contactId;
        u[r + D_TYPE_A] = job.shapeA.type;
        u[r + D_TYPE_B] = job.shapeB.type;
        writeXf(f, r + D_XF_A, job.xfA);
        writeXf(f, r + D_XF_B, job.xfB);
        writeGeom(world, f, u, r + D_GEOM_A, job.shapeA);
        writeGeom(world, f, u, r + D_GEOM_B, job.shapeB);
        u[r + D_CHILD] = job.contact.childIndex;
        u[r + D_MESH_SLOT] = job.meshSlot;
        u[r + D_FAST] = Number(job.isFast);
        u[r + D_OLD_COUNT] = job.contact.manifoldCount;
        u[r + D_CACHE_VALID] = Number(job.contact.kernelMeshCache !== null);
        if (job.meshSlot !== -1) {
            writeVec(f, r + D_LOWER, job.shapeB.aabb.lowerBound);
            writeVec(f, r + D_UPPER, job.shapeB.aabb.upperBound);
            if (job.contact.kernelMeshCache !== null)
                new Uint32Array(
                    buf,
                    k.meshCachePtr() + job.meshSlot * cacheWords * 4,
                    cacheWords,
                ).set(job.contact.kernelMeshCache);
        }
    }
    runPar(world.ecsState, ParKind.Contacts, jobCount, 0, 0, () => k.dispatchContacts(jobCount));
    const out = new Uint32Array(buf, k.dispatchOutPtr(), jobCount);
    // Read every result before allocating manifold blocks: growth can detach or overwrite transient columns.
    for (let i = 0; i < jobCount; ++i) {
        const job = jobs[i];
        results[i] = out[i];
        if (job.meshSlot === -1) continue;
        job.contact.kernelMeshCache ??= new Uint32Array(cacheWords);
        job.contact.kernelMeshCache.set(
            new Uint32Array(buf, k.meshCachePtr() + job.meshSlot * cacheWords * 4, cacheWords),
        );
        job.result ??= new Uint32Array(256 * MANIFOLD_STRIDE);
        job.materials ??= new Uint32Array(256 * 4);
        job.result.set(
            new Uint32Array(
                buf,
                k.meshOutputPtr() + job.meshSlot * 256 * MANIFOLD_STRIDE * 4,
                out[i] * MANIFOLD_STRIDE,
            ),
        );
        job.materials.set(
            new Uint32Array(buf, k.meshMaterialPtr() + job.meshSlot * 256 * 4 * 4, out[i] * 4),
        );
    }
    for (let i = 0; i < jobCount; ++i) finish(world, jobs[i], results[i]);
}
function recycle(world: WorldState, distance: number): void {
    refreshGeometry(world);
    const k = kernel(world.ecsState),
        contacts = world.awakeContacts,
        count = contacts.length;
    k.reserveRecycle(count);
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    const buf = k.memory.buffer;
    const u = new Uint32Array(buf, k.recyclePtr(), count * RECYCLE_STRIDE);
    const f = new Float32Array(buf, k.recyclePtr(), count * RECYCLE_STRIDE);
    for (let i = 0; i < count; ++i) {
        const contact = world.contacts[contacts[i]],
            r = i * RECYCLE_STRIDE;
        const bodyA = world.bodies[contact.edges[0].bodyId],
            bodyB = world.bodies[contact.edges[1].bodyId];
        const simA = getBodySim(world, bodyA),
            simB = getBodySim(world, bodyB);
        u[r + R_CONTACT] = contact.contactId;
        u[r + R_LOCAL_A] = bodyA.setIndex === SetType.Awake ? contact.bodySimIndexA : NULL_INDEX;
        u[r + R_LOCAL_B] = bodyB.setIndex === SetType.Awake ? contact.bodySimIndexB : NULL_INDEX;
        u[r + R_SHAPE_A] = contact.shapeIdA;
        u[r + R_SHAPE_B] = contact.shapeIdB;
        const fastMesh =
            (contact.flags & ContactFlags.simMeshContact) !== 0 &&
            ((simA.flags | simB.flags) & BodyFlags.isFast) !== 0;
        let bits = 0;
        if (
            !fastMesh &&
            distance > 0 &&
            (contact.flags & ContactFlags.relativeTransformValid) !== 0 &&
            (contact.flags & ContactFlags.contactRecycleFlag) !== 0
        )
            bits |= R_ELIGIBLE;
        if ((contact.flags & ContactFlags.simTouchingFlag) !== 0) bits |= R_WAS_TOUCHING;
        u[r + R_BITS] = bits;
        u[r + R_COUNT] = contact.manifoldCount;
        if (bodyA.setIndex !== SetType.Awake) {
            writeXf(f, r + R_FALLBACK_A, simA.transform);
            writeVec(f, r + R_FALLBACK_A + 7, simA.center);
            writeVec(
                f,
                r + R_FALLBACK_A + 10,
                bodyA.type === BodyType.Static ? zero : simA.maxExtent,
            );
        }
        if (bodyB.setIndex !== SetType.Awake) {
            writeXf(f, r + R_FALLBACK_B, simB.transform);
            writeVec(f, r + R_FALLBACK_B + 7, simB.center);
            writeVec(
                f,
                r + R_FALLBACK_B + 10,
                bodyB.type === BodyType.Static ? zero : simB.maxExtent,
            );
        }
    }
    const speculative = minf(distance, SPECULATIVE_DISTANCE);
    runPar(world.ecsState, ParKind.Recycle, count, distance, speculative, () =>
        k.dispatchRecycle(count, distance, speculative),
    );
    const out = new Uint32Array(buf, k.recycleOutPtr(), count);
    for (let i = 0; i < count; ++i) results[i] = out[i];
    for (let i = 0; i < count; ++i) {
        const contact = world.contacts[contacts[i]];
        if (results[i] === 0) continue;
        if (results[i] === 2) {
            contact.flags |= ContactFlags.simDisjoint;
            contact.flags &= ~ContactFlags.simTouchingFlag;
            stateChanges.push(contact.contactId);
            continue;
        }
        contact.flags |= ContactFlags.relativeTransformValid;
        collect(world, contact);
    }
}
function addNonTouchingContact(world: WorldState, contact: Contact): void {
    const set = world.solverSets[SetType.Awake];
    contact.colorIndex = NULL_INDEX;
    contact.localIndex = set.contactIndices.length;
    set.contactIndices.push(contact.contactId);
}
function removeNonTouchingContact(world: WorldState, setIndex: number, localIndex: number): void {
    const set = world.solverSets[setIndex];
    if (swapRemove(set.contactIndices, localIndex) !== NULL_INDEX)
        world.contacts[set.contactIndices[localIndex]].localIndex = localIndex;
}

/** Run narrowphase tasks for every awake contact, then commit serial touching transitions. */
export function collide(context: StepContext): void {
    const world = context.world;
    world.manifoldStore.refreshViews();
    stateChanges.length = 0;
    jobCount = 0;
    recycle(world, world.contactRecycleDistance);
    if (jobCount > 0) dispatch(world);
    stateChanges.sort((a, b) => a - b);
    const endEventArrayIndex = world.endEventArrayIndex,
        worldId = world.worldId;
    for (const contactId of stateChanges) {
        const contact = world.contacts[contactId];
        const shapeA = world.shapes[contact.shapeIdA],
            shapeB = world.shapes[contact.shapeIdB];
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
                    normalImpulse: 0,
                });
            }
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
                    normalImpulse: 0,
                });
            }
            const colorIndex = contact.colorIndex,
                localIndex = contact.localIndex;
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
