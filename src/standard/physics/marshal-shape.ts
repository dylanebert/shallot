import { Hulls, Shape as ShapeComponent, ShapeKind, ShapeMaterials } from "../../core/physics";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { makeShapeId } from "./api/config";
import type { EntityId } from "./common/ids";
import { defaultShapeDef, type Filter, type ShapeDef, type SurfaceMaterial } from "./common/types";
import { buildPlacedHull, compoundBuild, heightFieldBuild, meshBuild } from "./geometry-builds";
import type { HullData } from "./shapes/hull";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    shapeIsValid,
} from "./shapes/shape";
import type { WorldState } from "./world/world";

export function authoredShapeMaterial(world: World, eid: number): SurfaceMaterial {
    const shape = world.storage(ShapeComponent);
    return {
        friction: shape.friction.get(eid),
        restitution: shape.restitution.get(eid),
        rollingResistance: shape.rollingResistance.get(eid),
        tangentVelocity: [
            shape.tangentVelocity.x.get(eid),
            shape.tangentVelocity.y.get(eid),
            shape.tangentVelocity.z.get(eid),
        ],
        userMaterialId:
            (BigInt(shape.userMaterialIdHigh.get(eid) >>> 0) << 32n) |
            BigInt(shape.userMaterialIdLow.get(eid) >>> 0),
        customColor: shape.customColor.get(eid) >>> 0,
    };
}

export function authoredShapeFilter(world: World, eid: number): Filter {
    const shape = world.storage(ShapeComponent);
    return {
        categoryBits:
            (BigInt(shape.filterCategoryHigh.get(eid) >>> 0) << 32n) |
            BigInt(shape.filterCategoryLow.get(eid) >>> 0),
        maskBits:
            (BigInt(shape.filterMaskHigh.get(eid) >>> 0) << 32n) |
            BigInt(shape.filterMaskLow.get(eid) >>> 0),
        groupIndex: shape.filterGroupIndex.get(eid),
    };
}

function definition(world: World, eid: number): ShapeDef {
    const shape = world.storage(ShapeComponent);
    const defaults = defaultShapeDef();
    const materialSetId = shape.materialSet.get(eid);
    let materials: SurfaceMaterial[] | undefined;
    if (materialSetId !== -1) {
        const registered = world.resource(ShapeMaterials).get(materialSetId);
        if (!registered)
            throw new Error(`physics: Shape ${eid} names missing material set ${materialSetId}`);
        materials = registered.materials;
    }
    return {
        ...defaults,
        baseMaterial: authoredShapeMaterial(world, eid),
        materials,
        density: shape.density.get(eid),
        explosionScale: shape.explosionScale.get(eid),
        filter: authoredShapeFilter(world, eid),
        enableCustomFiltering: shape.enableCustomFiltering.get(eid) !== 0,
        isSensor: shape.isSensor.get(eid) !== 0,
        enableSensorEvents: shape.enableSensorEvents.get(eid) !== 0,
        enableContactEvents: shape.enableContactEvents.get(eid) !== 0,
        enableHitEvents: shape.enableHitEvents.get(eid) !== 0,
        enablePreSolveEvents: shape.enablePreSolveEvents.get(eid) !== 0,
        enableSpeculativeContact: shape.enableSpeculativeContact.get(eid) !== 0,
        invokeContactCreation: shape.invokeContactCreation.get(eid) !== 0,
        updateBodyMass: shape.updateBodyMass.get(eid) !== 0,
    };
}

/** Body-frame geometry uses the named body's authored scale, never a separate shape entity's placement. */
export function shapeBodyScale(world: World, eid: number) {
    const shape = world.storage(ShapeComponent);
    const body = shape.body.get(eid) || eid;
    if (!body) throw new Error(`physics: Shape ${eid} has no body for Transform.scale`);
    const scale = world.storage(Transform).scale;
    const x = scale.x.get(body),
        y = scale.y.get(body),
        z = scale.z.get(body);
    const kind = shape.kind.get(eid);
    if (![x, y, z].every((v) => Number.isFinite(v) && v !== 0))
        throw new Error(`physics: Shape ${eid} refuses zero or non-finite body Transform.scale`);
    if (kind === ShapeKind.HeightField && (x <= 0 || y <= 0 || z <= 0))
        throw new Error(`physics: HeightField Shape ${eid} requires positive body Transform.scale`);
    if (
        (kind === ShapeKind.Sphere || kind === ShapeKind.Capsule) &&
        (Math.abs(x) !== Math.abs(y) || Math.abs(x) !== Math.abs(z))
    )
        throw new Error(
            `physics: ${kind === ShapeKind.Sphere ? "Sphere" : "Capsule"} Shape ${eid} requires uniform body Transform.scale magnitudes`,
        );
    return { x, y, z };
}

export function shapeHullGeometry(world: World, eid: number): HullData {
    const shape = world.storage(ShapeComponent);
    const hulls = world.resource(Hulls);
    const id = shape.geometry.get(eid);
    const hull = hulls.get(id);
    if (!hull) throw new Error(`physics: Shape ${eid} names missing hull ${id}`);
    const s = shapeBodyScale(world, eid);
    return buildPlacedHull(
        world,
        hull,
        [shape.scale.x.get(eid), shape.scale.y.get(eid), shape.scale.z.get(eid)],
        [
            shape.hullPosition.x.get(eid),
            shape.hullPosition.y.get(eid),
            shape.hullPosition.z.get(eid),
        ],
        [
            shape.hullRotation.x.get(eid),
            shape.hullRotation.y.get(eid),
            shape.hullRotation.z.get(eid),
            shape.hullRotation.w.get(eid),
        ],
        [s.x, s.y, s.z],
    );
}

function localSphere(world: World, eid: number) {
    const shape = world.storage(ShapeComponent);
    const center = shape.sphereCenter;
    const s = shapeBodyScale(world, eid);
    return {
        center: {
            x: center.x.get(eid) * s.x,
            y: center.y.get(eid) * s.y,
            z: center.z.get(eid) * s.z,
        },
        radius: shape.radius.get(eid) * Math.abs(s.x),
    };
}

function localCapsule(world: World, eid: number) {
    const shape = world.storage(ShapeComponent);
    const s = shapeBodyScale(world, eid);
    return {
        center1: {
            x: shape.capsuleCenter1.x.get(eid) * s.x,
            y: shape.capsuleCenter1.y.get(eid) * s.y,
            z: shape.capsuleCenter1.z.get(eid) * s.z,
        },
        center2: {
            x: shape.capsuleCenter2.x.get(eid) * s.x,
            y: shape.capsuleCenter2.y.get(eid) * s.y,
            z: shape.capsuleCenter2.z.get(eid) * s.z,
        },
        radius: shape.radius.get(eid) * Math.abs(s.x),
    };
}

/** Adds one ECS Shape to its already-created Box3D body. */
export function marshalShape(
    world: World,
    state: WorldState,
    body: EntityId,
    eid: number,
): EntityId {
    const shape = world.storage(ShapeComponent);
    const kind = shape.kind.get(eid);
    const s = shapeBodyScale(world, eid);
    const def = definition(world, eid);
    const bodyIndex = body.index1 - 1;
    let shapeId: number | null;
    if (kind === ShapeKind.Sphere) {
        shapeId = createSphereShape(state, bodyIndex, def, localSphere(world, eid));
    } else if (kind === ShapeKind.Capsule) {
        shapeId = createCapsuleShape(state, bodyIndex, def, localCapsule(world, eid));
    } else if (kind === ShapeKind.Hull) {
        shapeId = createHullShape(state, bodyIndex, def, shapeHullGeometry(world, eid));
    } else if (kind === ShapeKind.Mesh) {
        const data = meshBuild(world, shape.geometry.get(eid));
        if (!data)
            throw new Error(`physics: Shape ${eid} names missing mesh ${shape.geometry.get(eid)}`);
        shapeId = createMeshShape(state, bodyIndex, def, data, {
            x: shape.scale.x.get(eid) * s.x,
            y: shape.scale.y.get(eid) * s.y,
            z: shape.scale.z.get(eid) * s.z,
        });
    } else if (kind === ShapeKind.HeightField) {
        const data = heightFieldBuild(world, shape.geometry.get(eid), s.x, s.y, s.z);
        if (!data)
            throw new Error(
                `physics: Shape ${eid} names missing height-field ${shape.geometry.get(eid)}`,
            );
        shapeId = createHeightFieldShape(state, bodyIndex, def, data);
    } else if (kind === ShapeKind.Compound) {
        const data = compoundBuild(world, shape.geometry.get(eid), s.x, s.y, s.z);
        if (!data)
            throw new Error(
                `physics: Shape ${eid} names missing compound ${shape.geometry.get(eid)}`,
            );
        shapeId = createCompoundShape(state, bodyIndex, def, data);
    } else {
        throw new Error(`physics: Shape ${eid} has unknown kind ${kind}`);
    }
    if (shapeId === null || !shapeIsValid(state, makeShapeId(state, shapeId)))
        throw new Error(`physics: Shape ${eid} could not be attached to its body`);
    return makeShapeId(state, shapeId);
}

/** Build the Box3D primitive values from the authored Shape columns. */
export function shapeGeometry(
    world: World,
    eid: number,
): {
    sphere: ReturnType<typeof localSphere>;
    capsule: ReturnType<typeof localCapsule>;
    scale: { x: number; y: number; z: number };
} {
    const shape = world.storage(ShapeComponent);
    return {
        sphere: localSphere(world, eid),
        capsule: localCapsule(world, eid),
        scale: { x: shape.scale.x.get(eid), y: shape.scale.y.get(eid), z: shape.scale.z.get(eid) },
    };
}
