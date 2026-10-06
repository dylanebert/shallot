// Box3D's collide tasks own contact updates; the ascending touch pass stays with graph/island owners.
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { f32, quat, vec3, type WorldTransform, xf } from "../common/math";
import { defaultSurfaceMaterial, ShapeType } from "../common/types";
import { readSimTransform } from "../kernel/bodycolumns";
import { type Kernel, kernel, ParKind, runPool, threads, workers } from "../kernel/kernel";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import {
    setArrayCount,
    setArrayGet,
    setArrayPush,
    setArrayRemove,
} from "../kernel/solversetcolumns";
import { shapeHullInnerRadius } from "../shapes/hull";
import { getShapeMaterial, getShapeMaterialCount, type Shape, shapeRadius } from "../shapes/shape";
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
import {
    DIR_BLOCK,
    DIR_STRIDE,
    M_POINT_COUNT,
    M_POINTS,
    MANIFOLD_STRIDE,
    POINT_STRIDE,
    writeContactMaterial,
} from "./manifoldstore";

const SIM_UPDATED = 0x02000000;
const zero = vec3.zero(),
    tangentA = vec3.zero(),
    tangentB = vec3.zero();
const poseA = xf.identity(),
    poseB = xf.identity(),
    childPose = xf.identity();
const materialA = defaultSurfaceMaterial(),
    materialB = defaultSurfaceMaterial(),
    mixedMaterial = defaultSurfaceMaterial();
const radiusReport = { radius: 0 };
let memoryU = new Uint32Array(0);
function memory(k: Kernel): void {
    if (memoryU.buffer !== k.memory.buffer) memoryU = new Uint32Array(k.memory.buffer);
}
function readRollingRadius(world: WorldState, shape: Shape, out: { radius: number }): void {
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Sphere:
            out.radius = shapeRadius(world, shape);
            break;
        case ShapeType.Capsule:
            out.radius = shapeRadius(world, shape);
            break;
        case ShapeType.Hull:
            out.radius = f32(0.25 * shapeHullInnerRadius(world, shape));
            break;
        default:
            out.radius = 0;
    }
}
function mixMesh(
    world: WorldState,
    id: number,
    shapeA: Shape,
    shapeB: Shape,
    xfA: WorldTransform,
    materialMap: number[] | null,
    triangleMaterials: number[] | undefined,
): void {
    getShapeMaterial(world, shapeB, materialB);
    vec3.copy(zero, tangentA);
    const materialCount = getShapeMaterialCount(world, shapeA);
    let mixedFriction = 0,
        mixedRestitution = 0;
    if (materialCount > 0) {
        let friction = 0,
            restitution = 0,
            samples = 0;
        const store = world.manifoldStore;
        const base = store.dirU[id * DIR_STRIDE + DIR_BLOCK] >>> 2;
        const count = contactField(world, id, ContactField.manifoldCount);
        for (let m = 0; m < count; ++m) {
            const o = base + m * MANIFOLD_STRIDE;
            for (let point = 0; point < store.poolU[o + M_POINT_COUNT]; ++point) {
                const triangle = store.poolI[o + M_POINTS + point * POINT_STRIDE + 12];
                let index =
                    shapeField(world, shapeA, ShapeField.type) === ShapeType.HeightField
                        ? world.shapeGeometry[shapeA].heightField!.materialIndices[triangle >> 1]
                        : triangleMaterials![triangle];
                if (materialMap !== null) index = materialMap[index];
                index = Math.max(0, Math.min(index, materialCount - 1));
                const material = world.shapeStore.readMaterialAt(shapeA, index, materialA);
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
        const material = getShapeMaterial(world, shapeA, materialA);
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
    readRollingRadius(world, shapeB, radiusReport);
    const radius =
        shapeField(world, shapeB, ShapeField.type) === ShapeType.Hull
            ? shapeHullInnerRadius(world, shapeB)
            : radiusReport.radius;
    mixedMaterial.friction = mixedFriction;
    mixedMaterial.restitution = mixedRestitution;
    mixedMaterial.rollingResistance = f32(materialB.rollingResistance * radius);
    quat.rotateOut(xfA.q, tangentA, tangentA);
    quat.rotateOut(poseB.q, materialB.tangentVelocity, tangentB);
    vec3.subOut(tangentA, tangentB, tangentA);
    vec3.copy(tangentA, mixedMaterial.tangentVelocity);
    writeContactMaterial(world.manifoldStore.dirF, id, mixedMaterial);
}
function mixContact(world: WorldState, id: number): void {
    const ownShapeA = contactField(world, id, ContactField.shapeIdA);
    const ownShapeB = contactField(world, id, ContactField.shapeIdB);
    let shapeA = ownShapeA,
        shapeB = ownShapeB;
    readSimTransform(world, getBodySim(world, contactBodyId(world, id, 0)), poseA);
    readSimTransform(world, getBodySim(world, contactBodyId(world, id, 1)), poseB);
    let xfA = poseA,
        xfB = poseB;
    let materialMap: number[] | null = null;
    let materialIndex = 0;
    let triangleMaterials =
        shapeField(world, shapeA, ShapeField.type) === ShapeType.Mesh
            ? world.shapeGeometry[shapeA].mesh!.data.materialIndices
            : undefined;
    readRollingRadius(world, shapeA, radiusReport);
    let radiusA = radiusReport.radius;
    readRollingRadius(world, shapeB, radiusReport);
    const radiusB = radiusReport.radius;
    if (shapeField(world, shapeA, ShapeField.type) === ShapeType.Compound) {
        const compound = world.shapeGeometry[shapeA].compound!;
        let index = contactField(world, id, ContactField.childIndex);
        let childType: ShapeType;
        if (index < compound.capsules.length) {
            const child = compound.capsules[index];
            childType = ShapeType.Capsule;
            radiusA = child.capsule.radius;
            materialIndex = child.materialIndex;
        } else if ((index -= compound.capsules.length) < compound.hulls.length) {
            const child = compound.hulls[index];
            childType = ShapeType.Hull;
            radiusA = f32(0.25 * child.hull.innerRadius);
            materialIndex = child.materialIndex;
            xfA = xf.mulOut(poseA, child.transform, childPose);
        } else if ((index -= compound.hulls.length) < compound.meshes.length) {
            const child = compound.meshes[index];
            childType = ShapeType.Mesh;
            radiusA = 0;
            materialMap = child.materialIndices;
            materialIndex = materialMap[0];
            triangleMaterials = child.meshData.materialIndices;
            xfA = xf.mulOut(poseA, child.transform, childPose);
        } else {
            const child = compound.spheres[index - compound.meshes.length];
            childType = ShapeType.Sphere;
            radiusA = child.sphere.radius;
            materialIndex = child.materialIndex;
        }
        if (
            (childType === ShapeType.Sphere &&
                shapeField(world, shapeB, ShapeField.type) !== ShapeType.Sphere) ||
            (childType === ShapeType.Capsule &&
                shapeField(world, shapeB, ShapeField.type) === ShapeType.Hull)
        ) {
            shapeA = ownShapeB;
            shapeB = ownShapeA;
            const pose = xfA;
            xfA = xfB;
            xfB = pose;
        }
    }
    if (contactField(world, id, ContactField.flags) & ContactFlags.simMeshContact) {
        mixMesh(world, id, shapeA, ownShapeB, xfA, materialMap, triangleMaterials);
        return;
    }
    const ownA = world.shapeStore.readMaterialAt(ownShapeA, materialIndex, materialA);
    getShapeMaterial(world, ownShapeB, materialB);
    const a = shapeA === ownShapeB ? materialB : ownA;
    const b = shapeA === ownShapeB ? ownA : materialB;
    mixedMaterial.friction = world.frictionCallback(
        a.friction,
        a.userMaterialId,
        b.friction,
        b.userMaterialId,
    );
    mixedMaterial.restitution = world.restitutionCallback(
        a.restitution,
        a.userMaterialId,
        b.restitution,
        b.userMaterialId,
    );
    mixedMaterial.rollingResistance =
        a.rollingResistance > 0 || b.rollingResistance > 0
            ? f32(Math.max(a.rollingResistance, b.rollingResistance) * Math.max(radiusA, radiusB))
            : 0;
    quat.rotateOut(xfA.q, a.tangentVelocity, tangentA);
    quat.rotateOut(xfB.q, b.tangentVelocity, tangentB);
    vec3.subOut(tangentA, tangentB, tangentA);
    vec3.copy(tangentA, mixedMaterial.tangentVelocity);
    writeContactMaterial(world.manifoldStore.dirF, id, mixedMaterial);
}
function removeNonTouchingContact(world: WorldState, index: number): void {
    const set = SetType.Awake;
    if (setArrayRemove(world, set, 0, index) !== NULL_INDEX)
        setContactField(world, setArrayGet(world, set, 0, index), ContactField.localIndex, index);
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
        const a = contactField(world, id, ContactField.shapeIdA);
        const b = contactField(world, id, ContactField.shapeIdB);
        const event = {
            shapeIdA: {
                index1: a + 1,
                world0: world.worldId,
                generation: shapeField(world, a, ShapeField.generation),
            },
            shapeIdB: {
                index1: b + 1,
                world0: world.worldId,
                generation: shapeField(world, b, ShapeField.generation),
            },
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
        const set = SetType.Awake;
        setContactField(world, id, ContactField.colorIndex, NULL_INDEX);
        setContactField(world, id, ContactField.localIndex, setArrayCount(world, set, 0));
        setArrayPush(world, set, 0, id);
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
    const count = k.awakeContactCount();
    const defaultMix =
        world.frictionCallback === defaultFrictionCallback &&
        world.restitutionCallback === defaultRestitutionCallback;
    const capacity = contactCapacity(world);
    k.reserveCollide(
        count,
        threads(world.ecsState),
        Number(defaultMix),
        world.contactRecycleDistance,
    );
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    memory(k);
    k.awakeContactCopy(k.collideListPtr());
    const pool = workers(world.ecsState);
    if (pool !== null && k.parBuild(ParKind.Contacts, count, pool.size + 1, 0))
        runPool(world.ecsState, pool, k.runMt, true);
    else k.dispatchContacts(count);
    world.manifoldStore.refreshViews();
    if (!defaultMix) {
        for (let i = 0; i < count; ++i) {
            const id = k.awakeContactGet(i);
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
