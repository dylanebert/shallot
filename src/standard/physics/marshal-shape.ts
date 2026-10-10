import {
    Compounds,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    Shape as ShapeComponent,
    ShapeKind,
    ShapeMaterials,
} from "../../core/physics";
import type { World } from "../../engine";
import {
    createHull,
    defaultShapeDef,
    type Filter,
    type ShapeDef,
    type Body as SolverBody,
    type Shape as SolverShape,
    type SurfaceMaterial,
} from "./api";
import type { CompoundData } from "./shapes/compound";
import type { HeightFieldData } from "./shapes/heightfield";
import type { HullData } from "./shapes/hull";
import type { MeshData } from "./shapes/mesh";

function geometryData<T>(
    registry: {
        name(id: number): string | undefined;
        get(name: string): { data: unknown } | undefined;
    },
    id: number,
): T | undefined {
    const name = registry.name(id);
    return name === undefined ? undefined : (registry.get(name)?.data as T | undefined);
}

export function authoredShapeMaterial(world: World, eid: number): SurfaceMaterial {
    const shape = world.storage(ShapeComponent);
    return {
        friction: shape.friction.get(eid),
        restitution: shape.restitution.get(eid),
        rollingResistance: shape.rollingResistance.get(eid),
        tangentVelocity: {
            x: shape.tangentVelocity.x.get(eid),
            y: shape.tangentVelocity.y.get(eid),
            z: shape.tangentVelocity.z.get(eid),
        },
        userMaterialId:
            (BigInt(shape.materialUserIdHigh.get(eid) >>> 0) << 32n) |
            BigInt(shape.materialUserIdLow.get(eid) >>> 0),
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

function definition(world: World, eid: number): Partial<ShapeDef> {
    const shape = world.storage(ShapeComponent);
    const defaults = defaultShapeDef();
    const materialSetId = shape.materialSet.get(eid);
    let materials: SurfaceMaterial[] | undefined;
    if (materialSetId !== 0) {
        const name = world.resource(ShapeMaterials).name(materialSetId - 1);
        const registered =
            name === undefined ? undefined : world.resource(ShapeMaterials).get(name);
        if (!registered)
            throw new Error(
                `physics: Shape ${eid} names missing material set ${materialSetId - 1}`,
            );
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

export function shapeHullGeometry(world: World, eid: number): HullData {
    const shape = world.storage(ShapeComponent);
    const hulls = world.resource(Hulls);
    const id = shape.geometry.get(eid);
    const name = hulls.name(id);
    const hull = name === undefined ? undefined : hulls.get(name);
    if (!hull) throw new Error(`physics: Shape ${eid} names missing hull ${id}`);
    const scale = shape.scale;
    const points = hull.verts.map(([x, y, z]) => ({
        x: x * scale.x.get(eid),
        y: y * scale.y.get(eid),
        z: z * scale.z.get(eid),
    }));
    const built = createHull(points, points.length);
    if (!built) throw new Error(`physics: hull for Shape ${eid} could not be built`);
    return built;
}

function localSphere(world: World, eid: number) {
    const field = world.storage(ShapeComponent).sphere;
    return {
        center: { x: field.x.get(eid), y: field.y.get(eid), z: field.z.get(eid) },
        radius: field.w.get(eid),
    };
}

function localCapsule(world: World, eid: number) {
    const shape = world.storage(ShapeComponent);
    return {
        center1: {
            x: shape.capsuleA.x.get(eid),
            y: shape.capsuleA.y.get(eid),
            z: shape.capsuleA.z.get(eid),
        },
        center2: {
            x: shape.capsuleB.x.get(eid),
            y: shape.capsuleB.y.get(eid),
            z: shape.capsuleB.z.get(eid),
        },
        radius: shape.capsuleB.w.get(eid),
    };
}

/** Adds one ECS Shape to its already-created Box3D body. */
export function marshalShape(world: World, body: SolverBody, eid: number): SolverShape {
    const shape = world.storage(ShapeComponent);
    const kind = shape.kind.get(eid);
    const def = definition(world, eid);
    let result: SolverShape;
    if (kind === ShapeKind.Sphere) {
        result = body.createSphere(def, localSphere(world, eid));
    } else if (kind === ShapeKind.Capsule) {
        result = body.createCapsule(def, localCapsule(world, eid));
    } else if (kind === ShapeKind.Hull) {
        result = body.createHull(def, shapeHullGeometry(world, eid));
    } else if (kind === ShapeKind.Mesh) {
        const data = geometryData<MeshData>(world.resource(PhysicsMeshes), shape.geometry.get(eid));
        if (!data)
            throw new Error(`physics: Shape ${eid} names missing mesh ${shape.geometry.get(eid)}`);
        result = body.createMesh(def, data, {
            x: shape.scale.x.get(eid),
            y: shape.scale.y.get(eid),
            z: shape.scale.z.get(eid),
        });
    } else if (kind === ShapeKind.HeightField) {
        const data = geometryData<HeightFieldData>(
            world.resource(HeightFields),
            shape.geometry.get(eid),
        );
        if (!data)
            throw new Error(
                `physics: Shape ${eid} names missing height-field ${shape.geometry.get(eid)}`,
            );
        result = body.createHeightField(def, data);
    } else if (kind === ShapeKind.Compound) {
        const data = geometryData<CompoundData>(world.resource(Compounds), shape.geometry.get(eid));
        if (!data)
            throw new Error(
                `physics: Shape ${eid} names missing compound ${shape.geometry.get(eid)}`,
            );
        result = body.createCompound(def, data);
    } else {
        throw new Error(`physics: Shape ${eid} has unknown kind ${kind}`);
    }
    if (!result.isValid())
        throw new Error(`physics: Shape ${eid} could not be attached to its body`);
    return result;
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
