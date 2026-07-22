// World-scoped physical constants and the length-unit base. Ported from Box3D's constants.h /
// core.h (Erin Catto, MIT). fround discipline per the README.
//
// Box3D scales its length-based constants by a settable length-unit (b3SetLengthUnitsPerMeter).
// The port fixes the unit at the default 1.0 — every fixture is generated at 1.0, and stages 1-6
// already bake `0.005` etc. as literals. A settable unit is a scoped follow-on: it would have to
// sweep every module's constant at once, and no stage exercises a non-unit world. So the derived
// constants below are the final, correct values for the locked scope, not a lesser stopgap.

import { f32 } from "./math";

/** Length units per meter. Fixed at 1.0 for the port; see the file header. */
export const LENGTH_UNITS_PER_METER = f32(1.0);

/// A small length used as a collision and constraint tolerance (B3_LINEAR_SLOP).
export const LINEAR_SLOP = f32(0.005 * LENGTH_UNITS_PER_METER);

/// Shapes are extended by this amount so speculative contacts can be created before touching
/// (B3_SPECULATIVE_DISTANCE).
export const SPECULATIVE_DISTANCE = f32(4.0 * LINEAR_SLOP);

/// Overlap queries report a hit within this slop of touching (B3_OVERLAP_SLOP).
export const OVERLAP_SLOP = f32(0.1 * LINEAR_SLOP);

/// Maximum points in a shape-cast proxy point cloud (B3_MAX_SHAPE_CAST_POINTS).
export const MAX_SHAPE_CAST_POINTS = 64;

/// Upper bound on the per-shape fat-AABB margin (B3_MAX_AABB_MARGIN).
export const MAX_AABB_MARGIN = f32(0.05 * LENGTH_UNITS_PER_METER);

/// The per-shape fat-AABB margin is this fraction of the shape extent, capped by MAX_AABB_MARGIN
/// (B3_AABB_MARGIN_FRACTION).
export const AABB_MARGIN_FRACTION = f32(0.125);

/// A contact is recycled rather than destroyed while the shapes stay within this distance
/// (B3_CONTACT_RECYCLE_DISTANCE).
export const CONTACT_RECYCLE_DISTANCE = f32(10.0 * LINEAR_SLOP);

/// Mesh manifolds are pushed apart by this rest offset to improve mesh collision quality, at the
/// cost of a small visual gap (B3_MESH_REST_OFFSET). PhysX/Unreal call it "rest offset".
export const MESH_REST_OFFSET = f32(1.0 * LINEAR_SLOP);

/// A large sanity bound on coordinates (B3_HUGE), 100 km at the default length unit. Single-precision
/// value; large-world double mode would widen it, but the port is single precision.
export const HUGE = f32(1.0e5 * LENGTH_UNITS_PER_METER);

/// An island falls asleep once every body in it stays below its sleep threshold for this long, in
/// seconds (B3_TIME_TO_SLEEP).
export const TIME_TO_SLEEP = f32(0.5);

/// A contact whose relative rotation stays within this dot-product of its cached pose is recycled
/// rather than re-collided (B3_CONTACT_RECYCLE_ANGULAR_DISTANCE ~= cos(7°)).
export const CONTACT_RECYCLE_ANGULAR_DISTANCE = f32(0.99240388);

/// The most contact points a single manifold can carry (B3_MAX_MANIFOLD_POINTS).
export const MAX_MANIFOLD_POINTS = 4;

/// Number of solver graph colors (B3_GRAPH_COLOR_COUNT); the last is the serial overflow color.
export const GRAPH_COLOR_COUNT = 24;

/// The overflow color index (B3_OVERFLOW_INDEX): constraints that cannot fit a graph color (a
/// dynamic body touching many others) spill here and are solved serially, scalar.
export const OVERFLOW_INDEX = GRAPH_COLOR_COUNT - 1;

/// Colors 0..DYNAMIC_COLOR_COUNT-1 hold dynamic-dynamic constraints (B3_DYNAMIC_COLOR_COUNT); the
/// remaining non-overflow colors are reserved for dynamic-static constraints, kept at a higher
/// solver priority (built from the high end) to reduce push-through tunneling.
export const DYNAMIC_COLOR_COUNT = GRAPH_COLOR_COUNT - 4;

/// Body name length including the null terminator in C (B3_BODY_NAME_LENGTH). Names are plain JS
/// strings here; kept only to truncate to the reference limit.
export const BODY_NAME_LENGTH = 18;

/// Default collision category and mask bits: all bits set (B3_DEFAULT_CATEGORY_BITS /
/// B3_DEFAULT_MASK_BITS = UINT64_MAX). Real u64 → bigint on the public filter types.
export const DEFAULT_CATEGORY_BITS = 0xffffffffffffffffn;
export const DEFAULT_MASK_BITS = 0xffffffffffffffffn;

/// The same all-bits default in the internal u32-halves form, for the bigint-free tree paths.
export const ALL_BITS_HI = 0xffffffff;
export const ALL_BITS_LO = 0xffffffff;

/// Solver-set roles (b3SetType). Sets 0-2 are permanent singletons created in this order at world
/// construction; every set index >= firstSleeping is a sleeping island group.
export const SetType = {
    Static: 0,
    Disabled: 1,
    Awake: 2,
    FirstSleeping: 3,
} as const;
export type SetType = (typeof SetType)[keyof typeof SetType];
