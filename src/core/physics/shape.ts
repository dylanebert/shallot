import { component, entity, f32, i32, u32, vec3 } from "../../engine";

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
        /** World-local geometry handle from Hulls, PhysicsMeshes, HeightFields or Compounds, selected by kind. */
        geometry: u32,
        /** Sphere center in body-local meters. */
        sphereCenter: vec3,
        /** First capsule hemisphere center in body-local meters. */
        capsuleCenter1: vec3,
        /** Second capsule hemisphere center in body-local meters. */
        capsuleCenter2: vec3,
        /** Sphere or capsule radius in meters. */
        radius: f32,
        /** Hull or mesh scale, multiplied per axis by the body's Transform.scale; default `[1, 1, 1]`.
         * A box is the unit cube at its full lengths. */
        scale: vec3,
        /** Shape density in kg/m³; changes apply at the next fixed sync. */
        density: f32,
        /** Non-dimensional scale used by world explosions; Box3D has no setter, so a change recreates the shape at the next fixed sync. */
        explosionScale: f32,
        /** Base material's Coulomb friction; changes apply at the next fixed sync. */
        friction: f32,
        /** Base material restitution; changes apply at the next fixed sync. */
        restitution: f32,
        /** Base material rolling resistance; changes apply at the next fixed sync. */
        rollingResistance: f32,
        /** Base material conveyor velocity in shape-local m/s; changes apply at the next fixed sync. */
        tangentVelocity: vec3,
        /** Low and high u32 halves of the base material's u64 user id; changes apply at the next fixed sync. */
        userMaterialIdLow: u32,
        userMaterialIdHigh: u32,
        /** Base material debug color; zero disables custom coloring. Changes apply at the next fixed sync. */
        customColor: u32,
        /** Low and high halves of the collision category bit set; changes apply at the next fixed sync. */
        filterCategoryLow: u32,
        filterCategoryHigh: u32,
        /** Low and high halves of the accepted collision mask; changes apply at the next fixed sync. */
        filterMaskLow: u32,
        filterMaskHigh: u32,
        /** Signed collision group, which overrides category/mask tests when nonzero; changes apply at the next fixed sync. */
        filterGroupIndex: i32,
        /** Whether custom filtering is enabled; Box3D has no setter, so a change recreates the shape at the next fixed sync. */
        enableCustomFiltering: u32,
        /** Whether this shape is a sensor; Box3D has no setter, so a change recreates the shape at the next fixed sync. */
        isSensor: u32,
        /** Whether sensor overlap events are enabled; changes apply at the next fixed sync. */
        enableSensorEvents: u32,
        /** Whether contact begin/end events are enabled; changes apply at the next fixed sync. */
        enableContactEvents: u32,
        /** Whether hit events are enabled; changes apply at the next fixed sync. */
        enableHitEvents: u32,
        /** Whether pre-solve events are enabled; changes apply at the next fixed sync. */
        enablePreSolveEvents: u32,
        /** Whether speculative contacts are enabled; Box3D has no setter, so a change recreates the shape at the next fixed sync. */
        enableSpeculativeContact: u32,
        /** Whether creating this shape requests contact creation; a change recreates the shape at the next fixed sync. */
        invokeContactCreation: u32,
        /** Whether creation, replacement, and density writes recompute body mass. */
        updateBodyMass: u32,
        /** ShapeMaterials handle plus one; zero means no per-triangle materials. */
        materialSet: u32,
    },
    {
        defaults: () => ({
            body: 0,
            kind: ShapeKind.Hull,
            geometry: 0,
            sphereCenter: [0, 0, 0],
            capsuleCenter1: [0, -0.5, 0],
            capsuleCenter2: [0, 0.5, 0],
            radius: 0.5,
            scale: [1, 1, 1],
            density: 1000,
            explosionScale: 1,
            friction: 0.6,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: [0, 0, 0],
            userMaterialIdLow: 0,
            userMaterialIdHigh: 0,
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

/** A Box3D surface material (`b3SurfaceMaterial`), as a mesh triangle or height-field cell uses it. */
export interface SurfaceMaterial {
    /** Coulomb friction. */
    friction: number;
    restitution: number;
    rollingResistance: number;
    /** Conveyor velocity in shape-local m/s. */
    tangentVelocity: [number, number, number];
    /** Application material id, passed to queries and material mixing. */
    userMaterialId: bigint;
    /** Debug color, low 24 bits RGB; zero disables custom coloring. */
    customColor: number;
}

/** Optional per-triangle surface materials referenced by Shape.materialSet (handle + 1). */
export interface ShapeMaterialSet {
    name: string;
    materials: SurfaceMaterial[];
}
