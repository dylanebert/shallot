// Box3D's collide tasks own contact updates; the ascending touch pass stays with graph/island owners.
import { NULL_INDEX, swapRemove } from "../common/array";
import { SetType } from "../common/constants";
import { f32, mulWorldTransforms, quat, vec3, type WorldTransform, xf } from "../common/math";
import { defaultSurfaceMaterial, ShapeType } from "../common/types";
import { readSimTransform } from "../kernel/bodycolumns";
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
import {
    ContactField,
    ContactFlags,
    contactBodyId,
    contactCapacity,
    contactField,
    destroyContact,
    setContactField,
} from "./contact";
import { readContactManifolds, writeContactMaterial } from "./manifoldstore";

const SIM_UPDATED = 0x02000000;
const zero = vec3.zero(),
    tangentA = vec3.zero(),
    tangentB = vec3.zero();
const poseA = xf.identity(),
    poseB = xf.identity();
const materialA = defaultSurfaceMaterial(),
    materialB = defaultSurfaceMaterial();
let memoryU = new Uint32Array(0);
function memory(k: Kernel): void {
    if (memoryU.buffer !== k.memory.buffer) memoryU = new Uint32Array(k.memory.buffer);
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
function mixMesh(
    world: WorldState,
    id: number,
    shapeA: Shape,
    shapeB: Shape,
    xfA: WorldTransform,
    materialMap: number[] | null,
): void {
    const materialsA = getShapeMaterials(world.ecsState, shapeA);
    getShapeMaterial(world, shapeB, materialB);
    vec3.copy(zero, tangentA);
    const materialCount = getShapeMaterialCount(world.ecsState, shapeA);
    let mixedFriction = 0,
        mixedRestitution = 0;
    if (materialCount > 0) {
        let friction = 0,
            restitution = 0,
            samples = 0;
        for (const m of readContactManifolds(world, id)) {
            for (const point of m.points) {
                let index =
                    shapeA.type === ShapeType.HeightField
                        ? shapeA.heightField!.materialIndices[point.triangleIndex >> 1]
                        : shapeA.mesh!.data.materialIndices[point.triangleIndex];
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
        shapeB.type === ShapeType.Hull ? shapeB.hull!.innerRadius : rollingRadius(shapeB);
    const rolling = f32(materialB.rollingResistance * radius);
    quat.rotateOut(xfA.q, tangentA, tangentA);
    quat.rotateOut(poseB.q, materialB.tangentVelocity, tangentB);
    vec3.subOut(tangentA, tangentB, tangentA);
    writeContactMaterial(
        world.manifoldStore.dirF,
        id,
        mixedFriction,
        mixedRestitution,
        rolling,
        tangentA.x,
        tangentA.y,
        tangentA.z,
    );
}
function mixContact(world: WorldState, id: number): void {
    const ownShapeA = world.shapes[contactField(world, id, ContactField.shapeIdA)];
    const ownShapeB = world.shapes[contactField(world, id, ContactField.shapeIdB)];
    let shapeA = ownShapeA,
        shapeB = ownShapeB;
    readSimTransform(getBodySim(world, world.bodies[contactBodyId(world, id, 0)]), poseA);
    readSimTransform(getBodySim(world, world.bodies[contactBodyId(world, id, 1)]), poseB);
    let xfA = poseA,
        xfB = poseB;
    let materialMap: number[] | null = null;
    if (shapeA.type === ShapeType.Compound) {
        const child = getCompoundChild(
            shapeA.compound!,
            contactField(world, id, ContactField.childIndex),
        );
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
            const pose = xfA;
            xfA = xfB;
            xfB = pose;
        }
    }
    if (contactField(world, id, ContactField.flags) & ContactFlags.simMeshContact) {
        mixMesh(world, id, shapeA, ownShapeB, xfA, materialMap);
        return;
    }
    const ownA =
        materialMap === null
            ? getShapeMaterial(world, ownShapeA, materialA)
            : getShapeMaterials(world.ecsState, ownShapeA)[materialMap[0]];
    getShapeMaterial(world, ownShapeB, materialB);
    const a = shapeA === ownShapeB ? materialB : ownA;
    const b = shapeA === ownShapeB ? ownA : materialB;
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
        id,
        friction,
        restitution,
        rolling,
        tangentA.x,
        tangentA.y,
        tangentA.z,
    );
}
function removeNonTouchingContact(world: WorldState, index: number): void {
    const set = world.solverSets[SetType.Awake];
    if (swapRemove(set.contactIndices, index) !== NULL_INDEX)
        setContactField(world, set.contactIndices[index], ContactField.localIndex, index);
}
function applyTouch(world: WorldState, id: number): void {
    const flags = contactField(world, id, ContactField.flags);
    if (flags & ContactFlags.simDisjoint) {
        destroyContact(world, id, false);
        return;
    }
    const started = (flags & ContactFlags.simStartedTouching) !== 0;
    const stopped = (flags & ContactFlags.simStoppedTouching) !== 0;
    if (!started && !stopped) return;
    if (flags & ContactFlags.contactEnableContactEvents) {
        const a = world.shapes[contactField(world, id, ContactField.shapeIdA)];
        const b = world.shapes[contactField(world, id, ContactField.shapeIdB)];
        const event = {
            shapeIdA: { index1: a.id + 1, world0: world.worldId, generation: a.generation },
            shapeIdB: { index1: b.id + 1, world0: world.worldId, generation: b.generation },
            contactId: {
                index1: id + 1,
                world0: world.worldId,
                generation: contactField(world, id, ContactField.generation),
            },
            normalImpulse: 0,
        };
        if (started) world.contactBeginEvents.push(event);
        else world.contactEndEvents[world.endEventArrayIndex].push(event);
    }
    if (started) {
        setContactField(
            world,
            id,
            ContactField.flags,
            (flags & ~ContactFlags.simStartedTouching) | ContactFlags.contactTouchingFlag,
        );
        linkContact(world, id);
        const old = contactField(world, id, ContactField.localIndex);
        addContactToGraph(world, id);
        removeNonTouchingContact(world, old);
    } else {
        setContactField(
            world,
            id,
            ContactField.flags,
            flags & ~(ContactFlags.simStoppedTouching | ContactFlags.contactTouchingFlag),
        );
        const color = contactField(world, id, ContactField.colorIndex);
        const local = contactField(world, id, ContactField.localIndex);
        unlinkContact(world, id);
        const set = world.solverSets[SetType.Awake];
        setContactField(world, id, ContactField.colorIndex, NULL_INDEX);
        setContactField(world, id, ContactField.localIndex, set.contactIndices.length);
        set.contactIndices.push(id);
        removeContactFromGraph(
            world,
            contactBodyId(world, id, 0),
            contactBodyId(world, id, 1),
            color,
            local,
            (flags & ContactFlags.simMeshContact) !== 0,
        );
    }
}
/** Update contacts in place, then apply the kernel's state bitset in ascending contact-id order. */
export function collide(context: StepContext): void {
    const world = context.world;
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const count = world.awakeContacts.length;
    const defaultMix =
        world.frictionCallback === defaultFrictionCallback &&
        world.restitutionCallback === defaultRestitutionCallback;
    const capacity = contactCapacity(world);
    k.reserveCollide(
        count,
        world.bodies.length,
        threads(world.ecsState),
        Number(defaultMix),
        world.contactRecycleDistance,
    );
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    memory(k);
    memoryU.set(world.awakeContacts, k.collideListPtr() >>> 2);
    const bodyBase = k.collideBodyPtr() >>> 2;
    // Body records still belong to TypeScript. Stage their read-only poses once per body, not per contact.
    for (const body of world.bodies) {
        if (body.id !== NULL_INDEX && body.contactCount > 0 && body.setIndex !== SetType.Disabled)
            memoryU[bodyBase + body.id] = bodyColumnIndex(world, body);
    }
    const pool = workers(world.ecsState);
    if (pool !== null && k.parBuild(ParKind.Contacts, count, pool.size + 1, 0))
        runPool(world.ecsState, pool, k.runMt, true);
    else k.dispatchContacts(count);
    world.manifoldStore.refreshViews();
    if (!defaultMix) {
        for (const id of world.awakeContacts) {
            const flags = contactField(world, id, ContactField.flags);
            if (flags & SIM_UPDATED && contactField(world, id, ContactField.manifoldCount) > 0)
                mixContact(world, id);
        }
    }
    const stateBase = k.contactStatePtr() >>> 2;
    for (let word = 0; word < Math.ceil(capacity / 32); ++word) {
        memory(k);
        let bits = memoryU[stateBase + word];
        while (bits !== 0) {
            const bit = 31 - Math.clz32(bits & -bits);
            bits = (bits & (bits - 1)) >>> 0;
            applyTouch(world, word * 32 + bit);
        }
    }
}
