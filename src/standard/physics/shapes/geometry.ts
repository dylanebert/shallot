import {
    type AABB,
    f32,
    type Mat3,
    maxf,
    minf,
    quat,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "../common/math";

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

// Working registers for the AABB out-variants below; never live across calls.
const aabbC1: Vec3 = { x: 0, y: 0, z: 0 };

/** {@link computeSphereAABB}, written into `o` — identical expression tree, no allocation. */
export function computeSphereAABBOut(shape: Sphere, transform: Transform, o: AABB): AABB {
    quat.rotateOut(transform.q, shape.center, aabbC1);
    vec3.addOut(aabbC1, transform.p, aabbC1);
    const r = shape.radius;
    o.lowerBound.x = f32(aabbC1.x - r);
    o.lowerBound.y = f32(aabbC1.y - r);
    o.lowerBound.z = f32(aabbC1.z - r);
    o.upperBound.x = f32(aabbC1.x + r);
    o.upperBound.y = f32(aabbC1.y + r);
    o.upperBound.z = f32(aabbC1.z + r);
    return o;
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

const _capsuleCenter1: Vec3 = { x: 0, y: 0, z: 0 };
const _capsuleCenter2: Vec3 = { x: 0, y: 0, z: 0 };

/** The same capsule bounds written into `out`, reusing the transformed endpoints. */
export function computeCapsuleAABBOut(shape: Capsule, transform: Transform, out: AABB): AABB {
    quat.rotateOut(transform.q, shape.center1, _capsuleCenter1);
    vec3.addOut(_capsuleCenter1, transform.p, _capsuleCenter1);
    quat.rotateOut(transform.q, shape.center2, _capsuleCenter2);
    vec3.addOut(_capsuleCenter2, transform.p, _capsuleCenter2);
    const r = shape.radius;
    out.lowerBound.x = f32(minf(_capsuleCenter1.x, _capsuleCenter2.x) - r);
    out.lowerBound.y = f32(minf(_capsuleCenter1.y, _capsuleCenter2.y) - r);
    out.lowerBound.z = f32(minf(_capsuleCenter1.z, _capsuleCenter2.z) - r);
    out.upperBound.x = f32(maxf(_capsuleCenter1.x, _capsuleCenter2.x) + r);
    out.upperBound.y = f32(maxf(_capsuleCenter1.y, _capsuleCenter2.y) + r);
    out.upperBound.z = f32(maxf(_capsuleCenter1.z, _capsuleCenter2.z) + r);
    return out;
}
