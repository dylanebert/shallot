import {
    type AABB,
    computeQuatBetweenUnitVectors,
    FLT_MIN,
    f32,
    type Mat3,
    mat3,
    maxf,
    minf,
    PI,
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

const FOUR_THIRDS = f32(4 / 3);

const massDensity = { density: 0 };
export function computeSphereMass(
    shape: Sphere,
    density: number,
    out: MassData = { mass: 0, center: vec3.zero(), inertia: mat3.zero() },
): MassData {
    massDensity.density = density;
    return computeSphereMassOut(shape, massDensity, out);
}

/** Compute from a reused density report into caller-owned mass storage. */
export function computeSphereMassOut(
    shape: Sphere,
    input: { density: number },
    out: MassData,
): MassData {
    const radius = shape.radius;
    const volume = f32(f32(f32(f32(FOUR_THIRDS * PI) * radius) * radius) * radius);
    const mass = f32(volume * input.density);
    // 0.4f is not exactly representable; fround the literal so the product matches C's f32 0.4f.
    const ixx = f32(f32(f32(f32(0.4) * mass) * radius) * radius);
    out.mass = mass;
    vec3.copy(shape.center, out.center);
    out.inertia.cx.x = ixx;
    out.inertia.cy.y = ixx;
    out.inertia.cz.z = ixx;
    out.inertia.cx.y =
        out.inertia.cx.z =
        out.inertia.cy.x =
        out.inertia.cy.z =
        out.inertia.cz.x =
        out.inertia.cz.y =
            0;
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

const massRotation = mat3.zero();
const massTranspose = mat3.zero();
const massProduct = mat3.zero();
const massDirection = vec3.zero();
const massQuat = { v: vec3.zero(), s: 1 };
const massAxis = { x: 0, y: 1, z: 0 };
export function computeCapsuleMass(
    shape: Capsule,
    density: number,
    out: MassData = { mass: 0, center: vec3.zero(), inertia: mat3.zero() },
): MassData {
    massDensity.density = density;
    return computeCapsuleMassOut(shape, massDensity, out);
}

/** Compute from a reused density report into caller-owned mass storage. */
export function computeCapsuleMassOut(
    shape: Capsule,
    input: { density: number },
    out: MassData,
): MassData {
    const c1 = shape.center1;
    const c2 = shape.center2;
    const r = shape.radius;

    // Cylinder
    const cylinderHeight = vec3.distance(c1, c2);
    const cylinderVolume = f32(f32(f32(PI * r) * r) * cylinderHeight);
    const cylinderMass = f32(cylinderVolume * input.density);

    // Sphere
    const sphereVolume = f32(f32(f32(f32(FOUR_THIRDS * PI) * r) * r) * r);
    const sphereMass = f32(sphereVolume * input.density);

    // Local accumulated inertia
    const inertia = out.inertia;
    const rr = f32(f32(3 * r) * r);
    const hh = f32(cylinderHeight * cylinderHeight);
    const cylinderX = f32(f32(cylinderMass * f32(rr + hh)) / 12);
    const cylinderY = f32(f32(f32(0.5 * cylinderMass) * r) * r);
    const sphereI = f32(f32(f32(f32(0.4) * sphereMass) * r) * r);
    inertia.cx.x = inertia.cz.z = f32(cylinderX + sphereI);
    inertia.cy.y = f32(cylinderY + sphereI);
    inertia.cx.y = inertia.cx.z = inertia.cy.x = inertia.cy.z = inertia.cz.x = inertia.cz.y = 0;

    // Steiner shift for the hemispheres offset from the cylinder center.
    const steinerShift = f32(
        f32(f32(0.125 * sphereMass) * f32(f32(3 * r) + f32(2 * cylinderHeight))) * cylinderHeight,
    );
    inertia.cx.x = f32(inertia.cx.x + steinerShift);
    inertia.cz.z = f32(inertia.cz.z + steinerShift);

    // Align capsule axis (y) with the segment direction.
    const rotation = massRotation;
    rotation.cx.x = rotation.cy.y = rotation.cz.z = 1;
    rotation.cx.y =
        rotation.cx.z =
        rotation.cy.x =
        rotation.cy.z =
        rotation.cz.x =
        rotation.cz.y =
            0;
    if (f32(cylinderHeight * cylinderHeight) > f32(1000 * FLT_MIN)) {
        vec3.subOut(c2, c1, massDirection);
        vec3.scaleOut(f32(1 / vec3.length(massDirection)), massDirection, massDirection);
        computeQuatBetweenUnitVectors(massAxis, massDirection, massQuat);
        mat3.fromQuatOut(massQuat, rotation);
    }
    out.mass = f32(sphereMass + cylinderMass);
    vec3.addOut(c1, c2, out.center);
    vec3.scaleOut(0.5, out.center, out.center);
    mat3.transposeOut(rotation, massTranspose);
    mat3.mulOut(inertia, massTranspose, massProduct);
    mat3.mulOut(rotation, massProduct, inertia);
    return out;
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
