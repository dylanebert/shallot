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
    xf,
} from "../common/math";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { readSimTransform } from "../kernel/bodycolumns";
import { bodyType, shapeBodyId } from "../kernel/filtercolumns";
import { readShapeAabb } from "../kernel/shapecolumns";

const dispatchBounds = { lowerBound: vec3.zero(), upperBound: vec3.zero() };

import {
    D_BODY_A,
    D_BODY_B,
    D_CHILD,
    D_CONTACT,
    D_DEFAULT_MIX,
    D_LOWER,
    D_MESH_SLOT,
    D_OLD_COUNT,
    D_RADIUS_A,
    D_SHAPE_A,
    D_SHAPE_B,
    D_UPPER,
    DISPATCH_STRIDE,
    R_BITS,
    R_CONTACT,
    R_COUNT,
    R_ELIGIBLE,
    R_LOCAL_A,
    R_LOCAL_B,
    R_MESH,
    R_SHAPE_A,
    R_SHAPE_B,
    R_STATIC_A,
    R_STATIC_B,
    R_WAS_TOUCHING,
    RECYCLE_STRIDE,
} from "../kernel/columns";
import { type Kernel, kernel, ParKind, runPool, threads, workers } from "../kernel/kernel";
import { bodyColumnIndex } from "../kernel/stagedbodies";
import { getCompoundChild } from "../shapes/compound";
import {
    getShapeMaterial,
    getShapeMaterialCount,
    getShapeMaterials,
    type Shape,
} from "../shapes/shape";
import type { StepContext } from "../solver/contactsolver";
import { addContactToGraph, removeContactFromGraph } from "../solver/graph";
import { getBodySim } from "../world/body";
import { linkContact, unlinkContact } from "../world/island";
import {
    defaultFrictionCallback,
    defaultRestitutionCallback,
    type WorldState,
} from "../world/world";
import { type Contact, ContactFlags, destroyContact, type Manifold } from "./contact";
import { MANIFOLD_STRIDE, writeContactMaterial } from "./manifoldstore";

const stateChanges: number[] = [];
const results: number[] = [];
const zero: Vec3 = { x: 0, y: 0, z: 0 };
const NO_MANIFOLDS: Manifold[] = [];
const poseA = xf.identity();
const poseB = xf.identity();
const tangentA = vec3.zero();
const tangentB = vec3.zero();
const materialA = defaultSurfaceMaterial();
const materialB = defaultSurfaceMaterial();
const ascending = (a: number, b: number) => a - b;
// Whole-memory views, re-made only when growth replaces the buffer, so a step mints none per phase.
let memoryU = new Uint32Array(0);
let memoryF = new Float32Array(0);
function memory(k: Kernel): void {
    if (memoryU.buffer === k.memory.buffer) return;
    memoryU = new Uint32Array(k.memory.buffer);
    memoryF = new Float32Array(k.memory.buffer);
}

type ContactJob = {
    contact: Contact;
    shapeA: Shape;
    shapeB: Shape;
    bodyA: number;
    bodyB: number;
    wasTouching: boolean;
    meshSlot: number;
    result: number;
    materials: Uint32Array | null;
};
const jobs: ContactJob[] = [];
let jobCount = 0;

function writeVec(f: Float32Array, o: number, v: Vec3): void {
    f[o] = v.x;
    f[o + 1] = v.y;
    f[o + 2] = v.z;
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
    const shapeA = world.shapes[contact.shapeIdA];
    const shapeB = world.shapes[contact.shapeIdB];
    const mesh = (contact.flags & ContactFlags.simMeshContact) !== 0;
    let job = jobs[jobCount];
    if (job === undefined) {
        job = {
            contact,
            shapeA,
            shapeB,
            bodyA: 0,
            bodyB: 0,
            wasTouching: false,
            meshSlot: -1,
            result: 0,
            materials: null,
        };
        jobs.push(job);
    }
    if (
        mesh &&
        (world.frictionCallback !== defaultFrictionCallback ||
            world.restitutionCallback !== defaultRestitutionCallback)
    )
        job.materials ??= new Uint32Array(256 * 4);
    job.contact = contact;
    job.shapeA = shapeA;
    job.shapeB = shapeB;
    job.bodyA = bodyColumnIndex(world, world.bodies[shapeBodyId(world, shapeA.id)]);
    job.bodyB = bodyColumnIndex(world, world.bodies[shapeBodyId(world, shapeB.id)]);
    job.wasTouching = (contact.flags & ContactFlags.simTouchingFlag) !== 0;
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
    getShapeMaterial(world, job.shapeB, materialB);
    vec3.copy(zero, tangentA);
    const materialCount = getShapeMaterialCount(world.ecsState, shapeA);
    let mixedFriction = 0,
        mixedRestitution = 0;
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
            mixedFriction = f32(inv * friction);
            mixedRestitution = f32(inv * restitution);
            vec3.scaleOut(inv, tangentA, tangentA);
        }
    } else {
        const material = materialsA[0];
        mixedFriction = world.frictionCallback(
            material.friction,
            material.userMaterialId,
            materialB.friction,
            materialB.userMaterialId,
        );
        mixedRestitution = world.restitutionCallback(
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
    const rolling = f32(materialB.rollingResistance * radius);
    quat.rotateOut(xfA.q, tangentA, tangentA);
    quat.rotateOut(poseB.q, materialB.tangentVelocity, tangentB);
    vec3.subOut(tangentA, tangentB, tangentA);
    writeContactMaterial(
        world.manifoldStore.dirF,
        contact.contactId,
        mixedFriction,
        mixedRestitution,
        rolling,
        tangentA.x,
        tangentA.y,
        tangentA.z,
    );
}
function finish(world: WorldState, job: ContactJob, count: number): void {
    const contact = job.contact;
    if (count === 0) {
        if (contact.manifoldCount > 0) world.manifoldStore.clear(contact.contactId);
        contact.manifolds = NO_MANIFOLDS;
        contact.manifoldCount = 0;
        contact.flags &= ~ContactFlags.simTouchingFlag;
        if (job.meshSlot !== -1) contact.flags &= ~ContactFlags.simEnableHitEvent;
        if (job.wasTouching) {
            contact.flags |= ContactFlags.simStoppedTouching;
            stateChanges.push(contact.contactId);
        }
        return;
    }
    if (job.meshSlot !== -1 || contact.manifoldCount === 0) {
        contact.manifolds = world.manifoldStore.importKernelManifolds(
            contact.contactId,
            count,
            job.result,
        );
        contact.manifoldCount = count;
    }
    if (
        world.frictionCallback !== defaultFrictionCallback ||
        world.restitutionCallback !== defaultRestitutionCallback
    ) {
        if (world.bodyStore.stale) world.bodyStore.refreshViews();
        let shapeA = job.shapeA,
            shapeB = job.shapeB;
        const simA = getBodySim(world, world.bodies[shapeBodyId(world, job.shapeA.id)]);
        const simB = getBodySim(world, world.bodies[shapeBodyId(world, job.shapeB.id)]);
        readSimTransform(simA, poseA);
        readSimTransform(simB, poseB);
        let xfA = poseA,
            xfB = poseB;
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
                xfA = mulWorldTransforms(poseA, child.transform);
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
            const ownA =
                materialMap === null
                    ? getShapeMaterial(world, job.shapeA, materialA)
                    : getShapeMaterials(world.ecsState, job.shapeA)[materialMap[0]];
            getShapeMaterial(world, job.shapeB, materialB);
            const a = shapeA === job.shapeB ? materialB : ownA;
            const b = shapeA === job.shapeB ? ownA : materialB;
            const friction = world.frictionCallback(
                a.friction,
                a.userMaterialId,
                b.friction,
                b.userMaterialId,
            );
            const restitution = world.restitutionCallback(
                a.restitution,
                a.userMaterialId,
                b.restitution,
                b.userMaterialId,
            );
            const rolling =
                a.rollingResistance > 0 || b.rollingResistance > 0
                    ? f32(
                          Math.max(a.rollingResistance, b.rollingResistance) *
                              Math.max(rollingRadius(shapeA), rollingRadius(shapeB)),
                      )
                    : 0;
            quat.rotateOut(xfA.q, a.tangentVelocity, tangentA);
            quat.rotateOut(xfB.q, b.tangentVelocity, tangentB);
            vec3.subOut(tangentA, tangentB, tangentA);
            writeContactMaterial(
                world.manifoldStore.dirF,
                contact.contactId,
                friction,
                restitution,
                rolling,
                tangentA.x,
                tangentA.y,
                tangentA.z,
            );
        }
    }
    if (job.shapeA.enableHitEvents || job.shapeB.enableHitEvents)
        contact.flags |= ContactFlags.simEnableHitEvent;
    else contact.flags &= ~ContactFlags.simEnableHitEvent;
    contact.flags |= ContactFlags.simTouchingFlag;
    if (!job.wasTouching) {
        contact.flags |= ContactFlags.simStartedTouching;
        stateChanges.push(contact.contactId);
    }
}
function dispatch(world: WorldState): void {
    const k = kernel(world.ecsState);
    let meshCount = 0;
    for (let i = 0; i < jobCount; ++i) {
        if (jobs[i].meshSlot === -1) continue;
        jobs[i].meshSlot = meshCount++;
        k.ensureMeshCache(jobs[i].contact.contactId);
    }
    k.reserveDispatch(jobCount, meshCount, threads(world.ecsState));
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    const buf = k.memory.buffer;
    memory(k);
    const f = memoryF,
        u = memoryU,
        base = k.dispatchPtr() >>> 2;
    for (let i = 0; i < jobCount; ++i) {
        const job = jobs[i],
            r = base + i * DISPATCH_STRIDE;
        u[r + D_CONTACT] = job.contact.contactId;
        u[r + D_DEFAULT_MIX] = Number(
            world.frictionCallback === defaultFrictionCallback &&
                world.restitutionCallback === defaultRestitutionCallback,
        );
        if (job.shapeA.type === ShapeType.Compound) {
            const child = getCompoundChild(job.shapeA.compound!, job.contact.childIndex);
            f[r + D_RADIUS_A] =
                child.type === ShapeType.Hull
                    ? f32(0.25 * child.hull!.innerRadius)
                    : child.type === ShapeType.Sphere
                      ? child.sphere!.radius
                      : child.type === ShapeType.Capsule
                        ? child.capsule!.radius
                        : 0;
        }
        u[r + D_SHAPE_A] = job.shapeA.id;
        u[r + D_SHAPE_B] = job.shapeB.id;
        u[r + D_BODY_A] = job.bodyA;
        u[r + D_BODY_B] = job.bodyB;
        u[r + D_CHILD] = job.contact.childIndex;
        u[r + D_MESH_SLOT] = job.meshSlot;
        u[r + D_OLD_COUNT] = job.contact.manifoldCount;
        if (job.meshSlot !== -1) {
            const bounds = readShapeAabb(world, job.shapeB.id, dispatchBounds);
            writeVec(f, r + D_LOWER, bounds.lowerBound);
            writeVec(f, r + D_UPPER, bounds.upperBound);
        }
    }
    const pool = workers(world.ecsState);
    if (pool !== null && k.parBuild(ParKind.Contacts, jobCount, pool.size + 1, 0, 0))
        runPool(world.ecsState, pool, k.runMt);
    else k.dispatchContacts(jobCount);
    const out = k.dispatchOutPtr() >>> 2;
    // Save byte offsets before allocation can detach views; collide scratch survives pool growth.
    for (let i = 0; i < jobCount; ++i) {
        const job = jobs[i];
        results[i] = u[out + i];
        job.result =
            job.meshSlot === -1
                ? (out + jobCount + i * MANIFOLD_STRIDE) * 4
                : k.meshOutputPtr() + job.meshSlot * 256 * MANIFOLD_STRIDE * 4;
        if (
            job.meshSlot !== -1 &&
            (world.frictionCallback !== defaultFrictionCallback ||
                world.restitutionCallback !== defaultRestitutionCallback)
        ) {
            job.materials!.set(
                new Uint32Array(
                    buf,
                    k.meshMaterialPtr() + job.meshSlot * 256 * 4 * 4,
                    results[i] * 4,
                ),
            );
        }
    }
    for (let i = 0; i < jobCount; ++i) finish(world, jobs[i], results[i]);
}
function recycle(world: WorldState): void {
    // Read here, not passed in: a double crossing a call is boxed every step.
    const distance = world.contactRecycleDistance;
    const k = kernel(world.ecsState),
        contacts = world.awakeContacts,
        count = contacts.length;
    k.reserveRecycle(count);
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    memory(k);
    const u = memoryU,
        base = k.recyclePtr() >>> 2;
    for (let i = 0; i < count; ++i) {
        const contact = world.contacts[contacts[i]],
            r = base + i * RECYCLE_STRIDE;
        const bodyA = world.bodies[contact.edges[0].bodyId],
            bodyB = world.bodies[contact.edges[1].bodyId];
        u[r + R_CONTACT] = contact.contactId;
        u[r + R_LOCAL_A] = bodyColumnIndex(world, bodyA);
        u[r + R_LOCAL_B] = bodyColumnIndex(world, bodyB);
        u[r + R_SHAPE_A] = contact.shapeIdA;
        u[r + R_SHAPE_B] = contact.shapeIdB;
        let bits = 0;
        if (bodyType(world, bodyA.id) === BodyType.Static) bits |= R_STATIC_A;
        if (bodyType(world, bodyB.id) === BodyType.Static) bits |= R_STATIC_B;
        if ((contact.flags & ContactFlags.simMeshContact) !== 0) bits |= R_MESH;
        if (
            distance > 0 &&
            (contact.flags & ContactFlags.relativeTransformValid) !== 0 &&
            (contact.flags & ContactFlags.contactRecycleFlag) !== 0
        )
            bits |= R_ELIGIBLE;
        if ((contact.flags & ContactFlags.simTouchingFlag) !== 0) bits |= R_WAS_TOUCHING;
        u[r + R_BITS] = bits;
        u[r + R_COUNT] = contact.manifoldCount;
    }
    const speculative = minf(distance, SPECULATIVE_DISTANCE);
    const pool = workers(world.ecsState);
    if (pool !== null && k.parBuild(ParKind.Recycle, count, pool.size + 1, distance, speculative))
        runPool(world.ecsState, pool, k.runMt);
    else k.dispatchRecycle(count, distance, speculative);
    const out = k.recycleOutPtr() >>> 2;
    for (let i = 0; i < count; ++i) results[i] = u[out + i];
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
    recycle(world);
    if (jobCount > 0) dispatch(world);
    stateChanges.sort(ascending);
    const endEventArrayIndex = world.endEventArrayIndex,
        worldId = world.worldId;
    for (let i = 0; i < stateChanges.length; ++i) {
        const contact = world.contacts[stateChanges[i]];
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
