import { component, entity, f32, i32, u32, vec4 } from "../../engine";

/** Box3D collision-shape kinds. A box is a scaled hull. */
export const ShapeKind = {
    Capsule: 0,
    Compound: 1,
    HeightField: 2,
    Hull: 3,
    Mesh: 4,
    Sphere: 5,
} as const;
export type ShapeKind = (typeof ShapeKind)[keyof typeof ShapeKind];

/** A Box3D shape definition and body-local collision geometry. A zero body reference means this entity. */
export const Shape = component(
    "Shape",
    {
        /** Body entity reference; zero attaches this shape to its own entity. */
        body: entity,
        /** Box3D shape kind. */
        kind: u32,
        /** Geometry registry id: Hulls, PhysicsMeshes, HeightFields or Compounds, selected by kind. */
        geometry: u32,
        /** Sphere center and radius in body-local meters. */
        sphere: vec4,
        /** First capsule endpoint in body-local meters. */
        capsuleA: vec4,
        /** Second capsule endpoint and radius in body-local meters. */
        capsuleB: vec4,
        /** Hull half-scale or mesh scale. */
        scale: vec4,
        /** Shape density in kg/m³; changes apply at the next fixed sync. */
        density: f32,
        /** Non-dimensional scale used by world explosions. */
        explosionScale: f32,
        /** Base material's Coulomb friction; changes apply at the next fixed sync. */
        friction: f32,
        /** Base material restitution; changes apply at the next fixed sync. */
        restitution: f32,
        /** Base material rolling resistance. */
        rollingResistance: f32,
        /** Base material conveyor velocity in shape-local m/s. */
        tangentVelocity: vec4,
        /** Low and high u32 halves of the base material's u64 user id. */
        materialUserIdLow: u32,
        materialUserIdHigh: u32,
        /** Base material debug color; zero disables custom coloring. */
        customColor: u32,
        /** Low and high halves of the collision category bit set. */
        filterCategoryLow: u32,
        filterCategoryHigh: u32,
        /** Low and high halves of the accepted collision mask. */
        filterMaskLow: u32,
        filterMaskHigh: u32,
        /** Signed collision group, which overrides category/mask tests when nonzero. */
        filterGroupIndex: i32,
        /** Whether custom filtering is enabled. Spawn-only; Box3D has no setter. */
        enableCustomFiltering: u32,
        /** Whether this shape is a sensor. Spawn-only; Box3D has no setter. */
        isSensor: u32,
        /** Whether sensor overlap events are enabled; changes apply at the next fixed sync. */
        enableSensorEvents: u32,
        /** Whether contact begin/end events are enabled; changes apply at the next fixed sync. */
        enableContactEvents: u32,
        /** Whether hit events are enabled; changes apply at the next fixed sync. */
        enableHitEvents: u32,
        /** Whether pre-solve events are enabled; changes apply at the next fixed sync. */
        enablePreSolveEvents: u32,
        /** Whether speculative contacts are enabled. Spawn-only; Box3D has no setter. */
        enableSpeculativeContact: u32,
        /** Whether creating this shape requests contact creation. Spawn-only. */
        invokeContactCreation: u32,
        /** Whether creation, replacement, and density writes recompute body mass. */
        updateBodyMass: u32,
        /** One-based ShapeMaterials registry id; zero means no per-triangle materials. */
        materialSet: u32,
    },
    {
        defaults: () => ({
            body: 0,
            kind: ShapeKind.Hull,
            geometry: 0,
            sphere: [0, 0, 0, 0.5],
            capsuleA: [0, -0.5, 0, 0],
            capsuleB: [0, 0.5, 0, 0.5],
            scale: [0.5, 0.5, 0.5, 0],
            density: 1000,
            explosionScale: 1,
            friction: 0.6,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: [0, 0, 0, 0],
            materialUserIdLow: 0,
            materialUserIdHigh: 0,
            customColor: 0,
            filterCategoryLow: 0xffffffff,
            filterCategoryHigh: 0xffffffff,
            filterMaskLow: 0xffffffff,
            filterMaskHigh: 0xffffffff,
            filterGroupIndex: 0,
            enableCustomFiltering: 0,
            isSensor: 0,
            enableSensorEvents: 0,
            enableContactEvents: 0,
            enableHitEvents: 0,
            enablePreSolveEvents: 0,
            enableSpeculativeContact: 1,
            invokeContactCreation: 1,
            updateBodyMass: 1,
            materialSet: 0,
        }),
    },
);

/** Named geometry registered by a game and referenced from Shape.geometry. */
export interface ShapeGeometry<T = unknown> {
    name: string;
    data: T;
}

/** Optional per-triangle surface materials referenced by Shape.materialSet (id + 1). */
export interface ShapeMaterialSet {
    name: string;
    materials: {
        friction: number;
        restitution: number;
        rollingResistance: number;
        tangentVelocity: { x: number; y: number; z: number };
        userMaterialId: bigint;
        customColor: number;
    }[];
}
