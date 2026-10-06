import { type AABB, f32, type Mat3, type Transform, type Vec3, vec3, xf } from "../common/math";

/** Mass, local center of mass, and inertia about that center (b3MassData). */
export type MassData = {
    mass: number;
    center: Vec3;
    inertia: Mat3;
};

/** A solid sphere (b3Sphere). */
export type Sphere = {
    center: Vec3;
    radius: number;
};

/** A solid capsule: two hemispheres of `radius` capping the segment center1→center2 (b3Capsule). */
export type Capsule = {
    center1: Vec3;
    center2: Vec3;
    radius: number;
};

// Callers pass f64 numbers; the C reference holds these as f32 struct fields, so every field must be
// rounded before storage or the solver arithmetic diverges (e.g. a radius of 0.3 is not f32-exact).
// Applied at each shape/compound-child storage boundary, mirroring the C float assignment.

/** Round a sphere's fields to f32 for storage. */
export function roundSphere(s: Sphere, out: Sphere = { center: vec3.zero(), radius: 0 }): Sphere {
    out.center.x = f32(s.center.x);
    out.center.y = f32(s.center.y);
    out.center.z = f32(s.center.z);
    out.radius = f32(s.radius);
    return out;
}

/** Round a capsule's fields to f32 for storage. */
export function roundCapsule(
    c: Capsule,
    out: Capsule = { center1: vec3.zero(), center2: vec3.zero(), radius: 0 },
): Capsule {
    out.center1.x = f32(c.center1.x);
    out.center1.y = f32(c.center1.y);
    out.center1.z = f32(c.center1.z);
    out.center2.x = f32(c.center2.x);
    out.center2.y = f32(c.center2.y);
    out.center2.z = f32(c.center2.z);
    out.radius = f32(c.radius);
    return out;
}

export function computeSphereAABB(shape: Sphere, transform: Transform): AABB {
    const center = xf.point(transform, shape.center);
    const r = shape.radius;
    const extent: Vec3 = { x: r, y: r, z: r };
    return { lowerBound: vec3.sub(center, extent), upperBound: vec3.add(center, extent) };
}

export function computeCapsuleAABB(shape: Capsule, transform: Transform): AABB {
    const r = shape.radius;
    const center1 = xf.point(transform, shape.center1);
    const center2 = xf.point(transform, shape.center2);
    const extent: Vec3 = { x: r, y: r, z: r };
    return {
        lowerBound: vec3.sub(vec3.min(center1, center2), extent),
        upperBound: vec3.add(vec3.max(center1, center2), extent),
    };
}
