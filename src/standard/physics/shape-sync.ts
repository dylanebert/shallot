import { PhysicsMeshes, ShapeKind, ShapeMaterials } from "../../core/physics";
import type { World } from "../../engine";
import type { Filter, SurfaceMaterial } from "./api";
import type { Shape as SolverShape } from "./api/shape";
import type { ShapeBinding } from "./authoring";
import { geometryIdentity, hullDatabaseIndex } from "./kernel/geocolumns";
import { kernel } from "./kernel/kernel";
import {
    authoredShapeFilter,
    authoredShapeMaterial,
    shapeGeometry,
    shapeHullGeometry,
} from "./marshal-shape";

function scalar(binding: ShapeBinding, eid: number, field: string): number {
    return (binding.storage[field] as { get(eid: number): number }).get(eid);
}
function changed(binding: ShapeBinding, mask: number, field: string): boolean {
    const index = binding.fieldIndices.get(field);
    return index !== undefined && (mask & (1 << index)) !== 0;
}
function vector(
    binding: ShapeBinding,
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
    binding: ShapeBinding,
    eid: number,
    values: Float64Array<ArrayBufferLike> = new Float64Array(RECREATION_FIELDS.length),
): Float64Array<ArrayBufferLike> {
    for (let i = 0; i < RECREATION_FIELDS.length; i++)
        values[i] = scalar(binding, eid, RECREATION_FIELDS[i]);
    return values;
}

function fieldDiffersFromSnapshot(
    binding: ShapeBinding,
    eid: number,
    field: string,
    previous: Float64Array | undefined,
): boolean {
    if (!previous) return true;
    const index = RECREATION_FIELD_INDEX.get(field);
    return index !== undefined && scalar(binding, eid, field) !== previous[index];
}
function anyChanged(binding: ShapeBinding, mask: number, fields: readonly string[]): boolean {
    for (const field of fields) if (changed(binding, mask, field)) return true;
    return false;
}
const MATERIAL_FIELDS = [
    "rollingResistance",
    "tangentVelocity",
    "materialUserIdLow",
    "materialUserIdHigh",
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
        Math.fround(a.tangentVelocity.x) === Math.fround(b.tangentVelocity.x) &&
        Math.fround(a.tangentVelocity.y) === Math.fround(b.tangentVelocity.y) &&
        Math.fround(a.tangentVelocity.z) === Math.fround(b.tangentVelocity.z) &&
        BigInt.asUintN(64, a.userMaterialId) === BigInt.asUintN(64, b.userMaterialId) &&
        a.customColor >>> 0 === b.customColor >>> 0
    );
}

function materialSet(
    world: World,
    binding: ShapeBinding,
    eid: number,
): SurfaceMaterial[] | undefined {
    const id = scalar(binding, eid, "materialSet");
    if (id === 0) return undefined;
    const name = world.resource(ShapeMaterials).name(id - 1);
    const set = name === undefined ? undefined : world.resource(ShapeMaterials).get(name);
    if (!set) throw new Error(`physics: Shape ${eid} names missing material set ${id - 1}`);
    return set.materials;
}

function expectedMeshScale(
    binding: ShapeBinding,
    eid: number,
): { x: number; y: number; z: number } {
    const scale = vector(binding, eid, "scale");
    const safe = (value: number) => (value >= 0 ? 1 : -1) * Math.max(Math.abs(value), 0.01);
    return { x: safe(scale[0]), y: safe(scale[1]), z: safe(scale[2]) };
}

/** Whether a marked definition field lacks a Box3D setter or needs a shape with different creation semantics. */
export function shapeNeedsRecreation(
    world: World,
    binding: ShapeBinding,
    eid: number,
    mask: number,
    previous: Float64Array | undefined,
    shape: SolverShape,
): boolean {
    const kind = scalar(binding, eid, "kind");
    if (kind !== shape.getType()) return true;
    if (
        changed(binding, mask, "explosionScale") &&
        fieldDiffersFromSnapshot(binding, eid, "explosionScale", previous)
    )
        return true;
    if (
        changed(binding, mask, "enableCustomFiltering") &&
        fieldDiffersFromSnapshot(binding, eid, "enableCustomFiltering", previous)
    )
        return true;
    if (
        changed(binding, mask, "enableSpeculativeContact") &&
        fieldDiffersFromSnapshot(binding, eid, "enableSpeculativeContact", previous)
    )
        return true;
    if (
        changed(binding, mask, "invokeContactCreation") &&
        fieldDiffersFromSnapshot(binding, eid, "invokeContactCreation", previous)
    )
        return true;
    if (changed(binding, mask, "isSensor")) {
        const next = scalar(binding, eid, "isSensor") !== 0;
        if (next !== shape.isSensor()) return true;
    }
    if (
        changed(binding, mask, "geometry") &&
        fieldDiffersFromSnapshot(binding, eid, "geometry", previous) &&
        (kind === ShapeKind.HeightField || kind === ShapeKind.Compound)
    )
        return true;
    if (
        changed(binding, mask, "materialSet") &&
        (kind === ShapeKind.Mesh || kind === ShapeKind.HeightField)
    ) {
        const materials = materialSet(world, binding, eid);
        const expectedCount = materials?.length || 1;
        if (shape.getMeshMaterialCount() !== expectedCount) return true;
    }
    return false;
}

/** Applies marked Box3D shape fields by comparing them with getters on the live shape. */
export function syncShapeFields(
    world: World,
    binding: ShapeBinding,
    eid: number,
    mask: number,
    shape: SolverShape,
): void {
    const kind = scalar(binding, eid, "kind");
    const materialSetChanged = changed(binding, mask, "materialSet");
    const hasMeshMaterials = kind === ShapeKind.Mesh || kind === ShapeKind.HeightField;
    const baseMaterialChanged = anyChanged(binding, mask, BASE_MATERIAL_FIELDS);
    const liveMaterials =
        hasMeshMaterials && (materialSetChanged || baseMaterialChanged)
            ? materialSet(world, binding, eid)
            : undefined;
    if (changed(binding, mask, "density")) {
        const next = scalar(binding, eid, "density");
        if (Number.isFinite(next) && next >= 0 && next !== shape.getDensity())
            shape.setDensity(next, scalar(binding, eid, "updateBodyMass") !== 0);
    }
    if (changed(binding, mask, "friction") && !liveMaterials?.length) {
        const next = scalar(binding, eid, "friction");
        if (Number.isFinite(next) && next >= 0 && next !== shape.getFriction())
            shape.setFriction(next);
    }
    if (changed(binding, mask, "restitution") && !liveMaterials?.length) {
        const next = scalar(binding, eid, "restitution");
        if (Number.isFinite(next) && next >= 0 && next !== shape.getRestitution())
            shape.setRestitution(next);
    }
    if (anyChanged(binding, mask, MATERIAL_FIELDS) && !liveMaterials?.length) {
        const next = authoredShapeMaterial(world, eid);
        if (!equalMaterial(next, shape.getSurfaceMaterial())) shape.setSurfaceMaterial(next);
    }
    if (anyChanged(binding, mask, FILTER_FIELDS)) {
        const next = authoredShapeFilter(world, eid);
        if (!equalFilter(next, shape.getFilter())) shape.setFilter(next);
    }
    if (changed(binding, mask, "enableSensorEvents")) {
        const next = scalar(binding, eid, "enableSensorEvents") !== 0;
        if (next !== shape.areSensorEventsEnabled()) shape.enableSensorEvents(next);
    }
    if (changed(binding, mask, "enableContactEvents")) {
        const next = scalar(binding, eid, "enableContactEvents") !== 0;
        if (next !== shape.areContactEventsEnabled()) shape.enableContactEvents(next);
    }
    if (changed(binding, mask, "enableHitEvents")) {
        const next = scalar(binding, eid, "enableHitEvents") !== 0;
        if (next !== shape.areHitEventsEnabled()) shape.enableHitEvents(next);
    }
    if (changed(binding, mask, "enablePreSolveEvents")) {
        const next = scalar(binding, eid, "enablePreSolveEvents") !== 0;
        if (next !== shape.arePreSolveEventsEnabled()) shape.enablePreSolveEvents(next);
    }

    if (kind === ShapeKind.Sphere && changed(binding, mask, "sphere")) {
        const sphere = shapeGeometry(world, eid).sphere;
        const current = shape.getSphere();
        if (
            Math.fround(sphere.center.x) !== current.center.x ||
            Math.fround(sphere.center.y) !== current.center.y ||
            Math.fround(sphere.center.z) !== current.center.z ||
            Math.fround(sphere.radius) !== current.radius
        )
            shape.setSphere(sphere);
    }
    if (
        kind === ShapeKind.Capsule &&
        (changed(binding, mask, "capsuleA") || changed(binding, mask, "capsuleB"))
    ) {
        const capsule = shapeGeometry(world, eid).capsule;
        const current = shape.getCapsule();
        if (
            Math.fround(capsule.center1.x) !== current.center1.x ||
            Math.fround(capsule.center1.y) !== current.center1.y ||
            Math.fround(capsule.center1.z) !== current.center1.z ||
            Math.fround(capsule.center2.x) !== current.center2.x ||
            Math.fround(capsule.center2.y) !== current.center2.y ||
            Math.fround(capsule.center2.z) !== current.center2.z ||
            Math.fround(capsule.radius) !== current.radius
        )
            shape.setCapsule(capsule);
    }
    if (
        kind === ShapeKind.Hull &&
        (changed(binding, mask, "geometry") || changed(binding, mask, "scale"))
    ) {
        const hull = shapeHullGeometry(world, eid);
        let nextReference: number | undefined;
        try {
            nextReference = hullDatabaseIndex(shape.world, hull);
        } catch {
            nextReference = undefined;
        }
        if (nextReference === undefined || shape.getGeometryReference() !== nextReference)
            shape.setHull(hull);
    }
    if (
        kind === ShapeKind.Mesh &&
        (changed(binding, mask, "geometry") || changed(binding, mask, "scale"))
    ) {
        const id = scalar(binding, eid, "geometry");
        const name = world.resource(PhysicsMeshes).name(id);
        const mesh = name === undefined ? undefined : world.resource(PhysicsMeshes).get(name)?.data;
        if (!mesh) throw new Error(`physics: Shape ${eid} names missing mesh ${id}`);
        const expectedScale = expectedMeshScale(binding, eid);
        const currentScale = shape.getGeometryScale();
        const identity = geometryIdentity(mesh as object);
        const reference =
            kernel(world).geometryDatabaseLookup(shape.world.worldId, 4, identity) >>> 0;
        if (
            reference !== shape.getGeometryReference() ||
            expectedScale.x !== currentScale.x ||
            expectedScale.y !== currentScale.y ||
            expectedScale.z !== currentScale.z
        )
            shape.setMesh(mesh as import("./shapes/mesh").MeshData, expectedScale);
    }
    if (materialSetChanged && hasMeshMaterials) {
        const materials = liveMaterials;
        const current = shape.getMaterials();
        const next = materials?.length ? materials : undefined;
        if (next) {
            for (let i = 0; i < next.length; i++)
                if (!equalMaterial(next[i], current[i])) shape.setMeshMaterial(next[i], i);
        } else if (current.length === 1) {
            const base = authoredShapeMaterial(world, eid);
            if (!equalMaterial(base, current[0])) shape.setMeshMaterial(base, 0);
        }
    }
}
