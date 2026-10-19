import { ShapeKind, ShapeMaterials } from "../../core/physics";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { type FieldBinding, fieldMarked } from "./authoring";
import type { EntityId } from "./common/ids";
import type { Filter, SurfaceMaterial } from "./common/types";
import { meshBuild } from "./geometry-builds";
import { geometryIdentity, hullDatabaseIndex } from "./kernel/geocolumns";
import { kernel } from "./kernel/kernel";
import { ShapeField, ShapeFlags, setShapeFlag, shapeField, shapeFlag } from "./kernel/shaperecords";
import {
    authoredShapeFilter,
    authoredShapeMaterial,
    shapeBodyScale,
    shapeGeometry,
    shapeHullGeometry,
} from "./marshal-shape";
import {
    getShapeDensity,
    getShapeFilter,
    getShapeGeometryReference,
    getShapeGeometryScale,
    getShapeMaterial,
    getShapeMaterialCount,
    getShapeMaterials,
    getShapeType,
    isSensorShape,
    readShapeCapsule,
    readShapeSphere,
    setShapeCapsule,
    setShapeDensity,
    setShapeFilter,
    setShapeFriction,
    setShapeHull,
    setShapeMesh,
    setShapeMeshMaterial,
    setShapeRestitution,
    setShapeSphere,
    setShapeSurfaceMaterial,
} from "./shapes/shape";
import { updateBodyMassData } from "./world/body";
import type { WorldState } from "./world/world";

function scalar(binding: FieldBinding, eid: number, field: string): number {
    return (binding.storage[field] as { get(eid: number): number }).get(eid);
}
function vector(
    binding: FieldBinding,
    eid: number,
    field: string,
): [number, number, number, number] {
    const value = binding.storage[field] as {
        x: { get(eid: number): number };
        y: { get(eid: number): number };
        z: { get(eid: number): number };
        w: { get(eid: number): number };
    };
    return [value.x.get(eid), value.y.get(eid), value.z.get(eid), value.w.get(eid)];
}
const RECREATION_FIELDS = [
    "explosionScale",
    "enableCustomFiltering",
    "enableSpeculativeContact",
    "invokeContactCreation",
    "geometry",
] as const;
const RECREATION_FIELD_INDEX: Map<string, number> = new Map(
    RECREATION_FIELDS.map((field, index) => [field, index]),
);

/** Captures only authored fields that require shape recreation and lack a live getter. */
export function captureShapeRecreationState(
    world: World,
    binding: FieldBinding,
    eid: number,
    values: Float64Array<ArrayBufferLike> = new Float64Array(RECREATION_FIELDS.length + 10),
): Float64Array<ArrayBufferLike> {
    for (let i = 0; i < RECREATION_FIELDS.length; i++)
        values[i] = scalar(binding, eid, RECREATION_FIELDS[i]);
    const s = shapeBodyScale(world, eid);
    values[RECREATION_FIELDS.length] = s.x;
    values[RECREATION_FIELDS.length + 1] = s.y;
    values[RECREATION_FIELDS.length + 2] = s.z;
    const position = vector(binding, eid, "hullPosition"),
        rotation = vector(binding, eid, "hullRotation");
    values.set(position.slice(0, 3), RECREATION_FIELDS.length + 3);
    values.set(rotation, RECREATION_FIELDS.length + 6);
    return values;
}

export function shapeBodyScaleChanged(
    world: World,
    body: number,
    previous: Float64Array | undefined,
): boolean {
    if (!previous) return true;
    const scale = world.storage(Transform).scale.column;
    const offset = body * 4;
    for (let i = 0; i < 3; i++)
        if (!Object.is(scale[offset + i] ?? 0, previous[RECREATION_FIELDS.length + i])) return true;
    return false;
}

function fieldDiffersFromSnapshot(
    binding: FieldBinding,
    eid: number,
    field: string,
    previous: Float64Array | undefined,
): boolean {
    if (!previous) return true;
    const index = RECREATION_FIELD_INDEX.get(field);
    return index !== undefined && scalar(binding, eid, field) !== previous[index];
}
function anyChanged(binding: FieldBinding, mask: number, fields: readonly string[]): boolean {
    for (const field of fields) if (fieldMarked(binding, mask, field)) return true;
    return false;
}
const MATERIAL_FIELDS = [
    "rollingResistance",
    "tangentVelocity",
    "userMaterialIdLow",
    "userMaterialIdHigh",
    "customColor",
] as const;
const FILTER_FIELDS = [
    "filterCategoryLow",
    "filterCategoryHigh",
    "filterMaskLow",
    "filterMaskHigh",
    "filterGroupIndex",
] as const;
const BASE_MATERIAL_FIELDS = ["friction", "restitution", ...MATERIAL_FIELDS];
const shapeMaterialScratch: SurfaceMaterial = {
    friction: 0,
    restitution: 0,
    rollingResistance: 0,
    tangentVelocity: [0, 0, 0],
    userMaterialId: 0n,
    customColor: 0,
};

function equalFilter(a: Filter, b: Filter): boolean {
    return (
        a.categoryBits === b.categoryBits &&
        a.maskBits === b.maskBits &&
        a.groupIndex === b.groupIndex
    );
}
function equalMaterial(a: SurfaceMaterial, b: SurfaceMaterial): boolean {
    return (
        Math.fround(a.friction) === Math.fround(b.friction) &&
        Math.fround(a.restitution) === Math.fround(b.restitution) &&
        Math.fround(a.rollingResistance) === Math.fround(b.rollingResistance) &&
        Math.fround(a.tangentVelocity[0]) === Math.fround(b.tangentVelocity[0]) &&
        Math.fround(a.tangentVelocity[1]) === Math.fround(b.tangentVelocity[1]) &&
        Math.fround(a.tangentVelocity[2]) === Math.fround(b.tangentVelocity[2]) &&
        BigInt.asUintN(64, a.userMaterialId) === BigInt.asUintN(64, b.userMaterialId) &&
        a.customColor >>> 0 === b.customColor >>> 0
    );
}

function materialSet(
    world: World,
    binding: FieldBinding,
    eid: number,
): SurfaceMaterial[] | undefined {
    const id = scalar(binding, eid, "materialSet");
    if (id === -1) return undefined;
    const set = world.resource(ShapeMaterials).get(id);
    if (!set) throw new Error(`physics: Shape ${eid} names missing material set ${id}`);
    return set.materials;
}

function expectedMeshScale(
    world: World,
    binding: FieldBinding,
    eid: number,
): { x: number; y: number; z: number } {
    const scale = vector(binding, eid, "scale");
    const safe = (value: number) => (value >= 0 ? 1 : -1) * Math.max(Math.abs(value), 0.01);
    const s = shapeBodyScale(world, eid);
    return { x: safe(scale[0] * s.x), y: safe(scale[1] * s.y), z: safe(scale[2] * s.z) };
}

/** Whether a marked definition field lacks a Box3D setter or needs a shape with different creation semantics. */
export function shapeNeedsRecreation(
    world: World,
    binding: FieldBinding,
    eid: number,
    mask: number,
    previous: Float64Array | undefined,
    shape: EntityId,
    state: WorldState,
    bodyScaleChanged = false,
): boolean {
    const record = shape.index1 - 1;
    const kind = scalar(binding, eid, "kind");
    if (kind !== getShapeType(state, record)) return true;
    if (kind === ShapeKind.Hull && anyChanged(binding, mask, ["hullPosition", "hullRotation"])) {
        const placement = [
            ...vector(binding, eid, "hullPosition").slice(0, 3),
            ...vector(binding, eid, "hullRotation"),
        ];
        if (
            !previous ||
            placement.some(
                (value, i) => !Object.is(value, previous[RECREATION_FIELDS.length + 3 + i]),
            )
        )
            return true;
    }
    if (bodyScaleChanged && (kind === ShapeKind.HeightField || kind === ShapeKind.Compound))
        return true;
    if (
        fieldMarked(binding, mask, "explosionScale") &&
        fieldDiffersFromSnapshot(binding, eid, "explosionScale", previous)
    )
        return true;
    if (
        fieldMarked(binding, mask, "enableCustomFiltering") &&
        fieldDiffersFromSnapshot(binding, eid, "enableCustomFiltering", previous)
    )
        return true;
    if (
        fieldMarked(binding, mask, "enableSpeculativeContact") &&
        fieldDiffersFromSnapshot(binding, eid, "enableSpeculativeContact", previous)
    )
        return true;
    if (
        fieldMarked(binding, mask, "invokeContactCreation") &&
        fieldDiffersFromSnapshot(binding, eid, "invokeContactCreation", previous)
    )
        return true;
    if (fieldMarked(binding, mask, "isSensor")) {
        const next = scalar(binding, eid, "isSensor") !== 0;
        if (next !== isSensorShape(state, record)) return true;
    }
    if (
        fieldMarked(binding, mask, "geometry") &&
        fieldDiffersFromSnapshot(binding, eid, "geometry", previous) &&
        (kind === ShapeKind.HeightField || kind === ShapeKind.Compound)
    )
        return true;
    if (
        fieldMarked(binding, mask, "materialSet") &&
        (kind === ShapeKind.Mesh || kind === ShapeKind.HeightField)
    ) {
        const materials = materialSet(world, binding, eid);
        const expectedCount = materials?.length || 1;
        if (getShapeMaterialCount(state, record) !== expectedCount) return true;
    }
    return false;
}

/** Applies marked Box3D shape fields by comparing them with the live shape record. */
export function syncShapeFields(
    world: World,
    state: WorldState,
    binding: FieldBinding,
    eid: number,
    mask: number,
    id: EntityId,
    bodyScaleChanged = false,
): void {
    const shape = id.index1 - 1;
    let geometryChanged = false;
    const kind = scalar(binding, eid, "kind");
    const materialSetChanged = fieldMarked(binding, mask, "materialSet");
    const hasMeshMaterials = kind === ShapeKind.Mesh || kind === ShapeKind.HeightField;
    const baseMaterialChanged = anyChanged(binding, mask, BASE_MATERIAL_FIELDS);
    const liveMaterials =
        hasMeshMaterials && (materialSetChanged || baseMaterialChanged)
            ? materialSet(world, binding, eid)
            : undefined;
    if (fieldMarked(binding, mask, "density")) {
        const next = scalar(binding, eid, "density");
        if (Number.isFinite(next) && next >= 0 && next !== getShapeDensity(state, shape))
            setShapeDensity(state, shape, next, scalar(binding, eid, "updateBodyMass") !== 0);
    }
    if (fieldMarked(binding, mask, "friction") && !liveMaterials?.length) {
        const next = scalar(binding, eid, "friction");
        if (
            Number.isFinite(next) &&
            next >= 0 &&
            next !== getShapeMaterial(state, shape, shapeMaterialScratch).friction
        )
            setShapeFriction(state, shape, next);
    }
    if (fieldMarked(binding, mask, "restitution") && !liveMaterials?.length) {
        const next = scalar(binding, eid, "restitution");
        if (
            Number.isFinite(next) &&
            next >= 0 &&
            next !== getShapeMaterial(state, shape, shapeMaterialScratch).restitution
        )
            setShapeRestitution(state, shape, next);
    }
    if (anyChanged(binding, mask, MATERIAL_FIELDS) && !liveMaterials?.length) {
        const next = authoredShapeMaterial(world, eid);
        if (!equalMaterial(next, getShapeMaterial(state, shape, shapeMaterialScratch)))
            setShapeSurfaceMaterial(state, shape, next);
    }
    if (anyChanged(binding, mask, FILTER_FIELDS)) {
        const next = authoredShapeFilter(world, eid);
        if (!equalFilter(next, getShapeFilter(state, shape))) setShapeFilter(state, shape, next);
    }
    if (fieldMarked(binding, mask, "enableSensorEvents")) {
        const next = scalar(binding, eid, "enableSensorEvents") !== 0;
        if (next !== shapeFlag(state, shape, ShapeFlags.enableSensorEvents))
            setShapeFlag(state, shape, ShapeFlags.enableSensorEvents, next);
    }
    if (fieldMarked(binding, mask, "enableContactEvents")) {
        const next = scalar(binding, eid, "enableContactEvents") !== 0;
        if (next !== shapeFlag(state, shape, ShapeFlags.enableContactEvents))
            setShapeFlag(state, shape, ShapeFlags.enableContactEvents, next);
    }
    if (fieldMarked(binding, mask, "enableHitEvents")) {
        const next = scalar(binding, eid, "enableHitEvents") !== 0;
        if (next !== shapeFlag(state, shape, ShapeFlags.enableHitEvents))
            setShapeFlag(state, shape, ShapeFlags.enableHitEvents, next);
    }
    if (fieldMarked(binding, mask, "enablePreSolveEvents")) {
        const next = scalar(binding, eid, "enablePreSolveEvents") !== 0;
        if (next !== shapeFlag(state, shape, ShapeFlags.enablePreSolveEvents))
            setShapeFlag(state, shape, ShapeFlags.enablePreSolveEvents, next);
    }

    if (
        kind === ShapeKind.Sphere &&
        (bodyScaleChanged ||
            fieldMarked(binding, mask, "sphereCenter") ||
            fieldMarked(binding, mask, "radius"))
    ) {
        const sphere = shapeGeometry(world, eid).sphere;
        const current = readShapeSphere(state, shape);
        if (
            Math.fround(sphere.center.x) !== current.center.x ||
            Math.fround(sphere.center.y) !== current.center.y ||
            Math.fround(sphere.center.z) !== current.center.z ||
            Math.fround(sphere.radius) !== current.radius
        ) {
            setShapeSphere(state, shape, sphere);
            geometryChanged = true;
        }
    }
    if (
        kind === ShapeKind.Capsule &&
        (bodyScaleChanged ||
            fieldMarked(binding, mask, "capsuleCenter1") ||
            fieldMarked(binding, mask, "capsuleCenter2") ||
            fieldMarked(binding, mask, "radius"))
    ) {
        const capsule = shapeGeometry(world, eid).capsule;
        const current = readShapeCapsule(state, shape);
        if (
            Math.fround(capsule.center1.x) !== current.center1.x ||
            Math.fround(capsule.center1.y) !== current.center1.y ||
            Math.fround(capsule.center1.z) !== current.center1.z ||
            Math.fround(capsule.center2.x) !== current.center2.x ||
            Math.fround(capsule.center2.y) !== current.center2.y ||
            Math.fround(capsule.center2.z) !== current.center2.z ||
            Math.fround(capsule.radius) !== current.radius
        ) {
            setShapeCapsule(state, shape, capsule);
            geometryChanged = true;
        }
    }
    if (
        kind === ShapeKind.Hull &&
        (bodyScaleChanged ||
            fieldMarked(binding, mask, "geometry") ||
            fieldMarked(binding, mask, "scale"))
    ) {
        const hull = shapeHullGeometry(world, eid);
        let nextReference: number | undefined;
        try {
            nextReference = hullDatabaseIndex(state, hull);
        } catch {
            nextReference = undefined;
        }
        if (
            nextReference === undefined ||
            getShapeGeometryReference(state, shape) !== nextReference
        ) {
            setShapeHull(state, shape, hull);
            geometryChanged = true;
        }
    }
    if (
        kind === ShapeKind.Mesh &&
        (bodyScaleChanged ||
            fieldMarked(binding, mask, "geometry") ||
            fieldMarked(binding, mask, "scale"))
    ) {
        const id = scalar(binding, eid, "geometry");
        const mesh = meshBuild(world, id);
        if (!mesh) throw new Error(`physics: Shape ${eid} names missing mesh ${id}`);
        const expectedScale = expectedMeshScale(world, binding, eid);
        const currentScale = getShapeGeometryScale(state, shape);
        const identity = geometryIdentity(mesh);
        const reference = kernel(world).geometryDatabaseLookup(state.worldId, 4, identity) >>> 0;
        const currentReference = getShapeGeometryReference(state, shape);
        if (
            reference !== currentReference ||
            expectedScale.x !== currentScale.x ||
            expectedScale.y !== currentScale.y ||
            expectedScale.z !== currentScale.z
        ) {
            setShapeMesh(state, shape, mesh, expectedScale);
            geometryChanged = true;
        }
    }
    if (geometryChanged && scalar(binding, eid, "updateBodyMass") !== 0)
        updateBodyMassData(state, shapeField(state, shape, ShapeField.bodyId));
    if (materialSetChanged && hasMeshMaterials) {
        const materials = liveMaterials;
        const current = getShapeMaterials(state, shape);
        const next = materials?.length ? materials : undefined;
        if (next) {
            for (let i = 0; i < next.length; i++)
                if (!equalMaterial(next[i], current[i]))
                    setShapeMeshMaterial(state, shape, next[i], i);
        } else if (current.length === 1) {
            const base = authoredShapeMaterial(world, eid);
            if (!equalMaterial(base, current[0])) setShapeMeshMaterial(state, shape, base, 0);
        }
    }
}
