import {
    type AABB,
    computeQuatBetweenUnitVectors,
    cylinderInertia,
    FLT_MIN,
    f32,
    type Mat3,
    mat3,
    maxf,
    minf,
    PI,
    quat,
    sphereInertia,
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
export const roundSphere = (s: Sphere): Sphere => ({
    center: vec3.round(s.center),
    radius: f32(s.radius),
});

/** Round a capsule's fields to f32 for storage. */
export const roundCapsule = (c: Capsule): Capsule => ({
    center1: vec3.round(c.center1),
    center2: vec3.round(c.center2),
    radius: f32(c.radius),
});

const FOUR_THIRDS = f32(4 / 3);

export function computeSphereMass(shape: Sphere, density: number): MassData {
    const radius = shape.radius;
    const volume = f32(f32(f32(f32(FOUR_THIRDS * PI) * radius) * radius) * radius);
    const mass = f32(volume * density);
    // 0.4f is not exactly representable; fround the literal so the product matches C's f32 0.4f.
    const ixx = f32(f32(f32(f32(0.4) * mass) * radius) * radius);
    return { mass, center: shape.center, inertia: mat3.diagonal(ixx, ixx, ixx) };
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

export function computeCapsuleMass(shape: Capsule, density: number): MassData {
    const c1 = shape.center1;
    const c2 = shape.center2;
    const r = shape.radius;

    // Cylinder
    const cylinderHeight = vec3.distance(c1, c2);
    const cylinderVolume = f32(f32(f32(PI * r) * r) * cylinderHeight);
    const cylinderMass = f32(cylinderVolume * density);

    // Sphere
    const sphereVolume = f32(f32(f32(f32(FOUR_THIRDS * PI) * r) * r) * r);
    const sphereMass = f32(sphereVolume * density);

    // Local accumulated inertia
    const inertia = mat3.add(
        cylinderInertia(cylinderMass, r, cylinderHeight),
        sphereInertia(sphereMass, r),
    );

    // Steiner shift for the hemispheres offset from the cylinder center.
    const steinerShift = f32(
        f32(f32(0.125 * sphereMass) * f32(f32(3 * r) + f32(2 * cylinderHeight))) * cylinderHeight,
    );
    inertia.cx.x = f32(inertia.cx.x + steinerShift);
    inertia.cz.z = f32(inertia.cz.z + steinerShift);

    // Align capsule axis (y) with the segment direction.
    let rotation = mat3.identity();
    if (f32(cylinderHeight * cylinderHeight) > f32(1000 * FLT_MIN)) {
        const direction = vec3.normalize(vec3.sub(c2, c1));
        const q = computeQuatBetweenUnitVectors(vec3.axisY(), direction);
        rotation = mat3.fromQuat(q);
    }

    const mass = f32(sphereMass + cylinderMass);
    const center = vec3.scale(0.5, vec3.add(c1, c2));

    return {
        mass,
        center,
        inertia: mat3.mul(rotation, mat3.mul(inertia, mat3.transpose(rotation))),
    };
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
