// Shape geometry: mass properties, bounding boxes, and low-level queries for the primitive shapes.
// Ported op-for-op from Box3D's sphere.c / capsule.c (Erin Catto, MIT). fround discipline per
// the README.

import { LINEAR_SLOP, OVERLAP_SLOP } from "./core";
import {
    type CastOutput,
    type DistanceInput,
    emptyCache,
    emptyCastOutput,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeCastPairInput,
    type ShapeProxy,
    shapeCast,
    shapeDistance,
} from "./distance";
import {
    type AABB,
    clampf,
    computeQuatBetweenUnitVectors,
    cylinderInertia,
    FLT_EPSILON,
    FLT_MIN,
    f32,
    getLengthAndNormalize,
    type Mat3,
    mat3,
    PI,
    type Plane,
    pointToSegmentDistance,
    quat,
    segmentDistance,
    sphereInertia,
    type Transform,
    type Vec3,
    vec3,
    xf,
} from "./math";
import type { PlaneResult } from "./mover";

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
const aabbC2: Vec3 = { x: 0, y: 0, z: 0 };

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

/** AABB enclosing a sphere swept between two poses (b3ComputeSweptSphereAABB). */
export function computeSweptSphereAABB(shape: Sphere, xf1: Transform, xf2: Transform): AABB {
    const r = shape.radius;
    const extent: Vec3 = { x: r, y: r, z: r };
    const center1 = xf.point(xf1, shape.center);
    const center2 = xf.point(xf2, shape.center);
    return {
        lowerBound: vec3.sub(vec3.min(center1, center2), extent),
        upperBound: vec3.add(vec3.max(center1, center2), extent),
    };
}

/** AABB enclosing a capsule swept between two poses (b3ComputeSweptCapsuleAABB). */
export function computeSweptCapsuleAABB(shape: Capsule, xf1: Transform, xf2: Transform): AABB {
    const r = shape.radius;
    const extent: Vec3 = { x: r, y: r, z: r };
    const a = xf.point(xf1, shape.center1);
    const b = xf.point(xf1, shape.center2);
    const c = xf.point(xf2, shape.center1);
    const d = xf.point(xf2, shape.center2);
    return {
        lowerBound: vec3.sub(vec3.min(vec3.min(a, b), vec3.min(c, d)), extent),
        upperBound: vec3.add(vec3.max(vec3.max(a, b), vec3.max(c, d)), extent),
    };
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

/** {@link computeCapsuleAABB}, written into `o` — identical expression tree, no allocation. */
export function computeCapsuleAABBOut(shape: Capsule, transform: Transform, o: AABB): AABB {
    const r = shape.radius;
    quat.rotateOut(transform.q, shape.center1, aabbC1);
    vec3.addOut(aabbC1, transform.p, aabbC1);
    quat.rotateOut(transform.q, shape.center2, aabbC2);
    vec3.addOut(aabbC2, transform.p, aabbC2);
    vec3.minOut(aabbC1, aabbC2, o.lowerBound);
    vec3.maxOut(aabbC1, aabbC2, o.upperBound);
    o.lowerBound.x = f32(o.lowerBound.x - r);
    o.lowerBound.y = f32(o.lowerBound.y - r);
    o.lowerBound.z = f32(o.lowerBound.z - r);
    o.upperBound.x = f32(o.upperBound.x + r);
    o.upperBound.y = f32(o.upperBound.y + r);
    o.upperBound.z = f32(o.upperBound.z + r);
    return o;
}

// --- queries --------------------------------------------------------------------------------

/** True if `proxy` (in identity frame) is within overlap slop of the sphere (b3OverlapSphere). */
export function overlapSphere(
    shape: Sphere,
    shapeTransform: Transform,
    proxy: ShapeProxy,
): boolean {
    const input: DistanceInput = {
        proxyA: { points: [shape.center], count: 1, radius: shape.radius },
        proxyB: proxy,
        transform: xf.invMul(shapeTransform, xf.identity()),
        useRadii: true,
    };
    const output = shapeDistance(input, emptyCache());
    return output.distance < OVERLAP_SLOP;
}

/**
 * Ray vs sphere, in the sphere's frame (b3RayCastSphere).
 * Precision Improvements for Ray / Sphere Intersection, Ray Tracing Gems 2019.
 */
export function rayCastSphere(shape: Sphere, input: RayCastInput): CastOutput {
    const output = emptyCastOutput();

    const p = shape.center;

    // Shift ray so sphere center is the origin
    const s = vec3.sub(input.origin, p);

    const r = shape.radius;
    const rr = f32(r * r);

    const { v: d, length } = getLengthAndNormalize(input.translation);
    if (length === 0) {
        // zero length ray
        if (vec3.lengthSq(s) < rr) {
            output.point = input.origin;
            output.hit = true;
        }
        return output;
    }

    // Closest point on ray to origin: solve dot(s + t * d, d) = 0
    const t = -vec3.dot(s, d);
    const c = vec3.mulAdd(s, t, d);
    const cc = vec3.dot(c, c);

    if (cc > rr) {
        // closest point is outside the sphere
        return output;
    }

    const h = f32(Math.sqrt(f32(rr - cc)));
    const fraction = f32(t - h);

    if (fraction < 0 || f32(input.maxFraction * length) < fraction) {
        // intersection is outside the range of the ray segment
        if (vec3.lengthSq(s) < rr) {
            output.point = input.origin;
            output.hit = true;
        }
        return output;
    }

    const hitPoint = vec3.mulAdd(s, fraction, d);

    output.fraction = f32(fraction / length);
    if (output.fraction > input.maxFraction) {
        output.fraction = input.maxFraction;
    }

    output.normal = vec3.normalize(hitPoint);
    output.point = vec3.mulAdd(p, shape.radius, output.normal);
    output.hit = true;

    return output;
}

/** Shape cast against a sphere (b3ShapeCastSphere). */
export function shapeCastSphere(sphere: Sphere, input: ShapeCastInput): CastOutput {
    const pairInput: ShapeCastPairInput = {
        proxyA: { points: [sphere.center], count: 1, radius: sphere.radius },
        proxyB: input.proxy,
        transform: xf.identity(),
        translationB: input.translation,
        maxFraction: input.maxFraction,
        canEncroach: input.canEncroach,
    };
    return shapeCast(pairInput);
}

/** True if `proxy` (in identity frame) is within overlap slop of the capsule (b3OverlapCapsule). */
export function overlapCapsule(
    shape: Capsule,
    shapeTransform: Transform,
    proxy: ShapeProxy,
): boolean {
    const input: DistanceInput = {
        proxyA: { points: [shape.center1, shape.center2], count: 2, radius: shape.radius },
        proxyB: proxy,
        transform: xf.invMul(shapeTransform, xf.identity()),
        useRadii: true,
    };
    const output = shapeDistance(input, emptyCache());
    return output.distance < OVERLAP_SLOP;
}

/** Ray vs capsule, in the capsule's frame (b3RayCastCapsule). */
export function rayCastCapsule(shape: Capsule, input: RayCastInput): CastOutput {
    const c1 = shape.center1;
    const c2 = shape.center2;
    const r = shape.radius;

    const output = emptyCastOutput();

    const d = vec3.sub(c2, c1);

    // Fall back to sphere if the capsule is short
    const tol = f32(0.01 * LINEAR_SLOP);
    const lengthSquared = vec3.lengthSq(d);
    if (lengthSquared < f32(tol * tol)) {
        const sphereCenter = vec3.scale(0.5, vec3.add(shape.center1, shape.center2));
        return rayCastSphere({ center: sphereCenter, radius: shape.radius }, input);
    }

    // Vector from first center to ray origin.
    const s = vec3.sub(input.origin, c1);

    // Capsule axis
    const length = f32(Math.sqrt(lengthSquared));
    const axis = vec3.scale(f32(1 / length), d);

    // Project ray origin onto capsule axis.
    const u = vec3.dot(s, axis);

    // Closest point on infinite capsule axis, relative to c1.
    const c = vec3.scale(u, axis);

    // Vector from closest point to ray origin
    const sc = vec3.sub(s, c);

    // Squared distance from ray origin to capsule axis
    const sc2 = vec3.lengthSq(sc);

    // Is the ray origin within the infinite cylinder along the capsule axis?
    if (sc2 < f32(r * r)) {
        // Clamped barycentric coordinate of ray origin projected onto capsule axis.
        const uClamped = clampf(u, 0, length);

        // The closest point on the bounded capsule segment, relative to c1.
        const cp = vec3.scale(uClamped, axis);

        // Vector from ray origin to closest point on segment.
        const scp = vec3.sub(s, cp);

        // Is the ray origin within the capsule?
        if (vec3.lengthSq(scp) < f32(r * r)) {
            output.hit = true;
            output.point = input.origin;
            return output;
        }

        // The ray can hit an endcap.
        return rayCastSphere({ center: vec3.add(c1, cp), radius: r }, input);
    }

    // Ray axis. A zero length ray reaching here starts outside the capsule, so it misses.
    const dr = input.translation;
    const { v: rayAxis, length: rayLength } = getLengthAndNormalize(dr);
    if (rayLength === 0) {
        return output;
    }

    // Barycentric coordinate of ray end point.
    const v = f32(u + f32(input.maxFraction * vec3.dot(dr, axis)));

    // Early out: does the projected ray fall outside the capsule?
    if ((u < -r && v < -r) || (f32(length + r) < u && f32(length + r) < v)) {
        return output;
    }

    // Closest point between the ray segment and the capsule segment (RTCD 5.1.9).
    const a1 = axis;
    const a2 = rayAxis;
    const a12 = vec3.dot(a1, a2);

    // Ray distance to the near intersection with the infinite cylinder. Length units.
    let tr: number;

    const det = f32(1 - f32(a12 * a12));
    if (det < FLT_EPSILON) {
        // Ray nearly parallel to the axis: solve the 2D ray-versus-circle problem.
        const perp = vec3.mulSub(a2, a12, a1);
        const perp2 = vec3.lengthSq(perp);
        const beta = vec3.dot(sc, perp);
        const gamma = f32(sc2 - f32(r * r));
        const disc = f32(f32(beta * beta) - f32(perp2 * gamma));

        // Casting away from the axis, or the perpendicular gap never closes to the radius.
        if (beta >= 0 || disc < 0) {
            return output;
        }

        // Quadratic near root, alternate form to avoid cancellation as the ray nears parallel.
        tr = f32(gamma / f32(-beta + f32(Math.sqrt(disc))));
    } else {
        // Ray and capsule axes are not parallel.
        const invDet = f32(1 / det);
        const sa1 = u;
        const sa2 = vec3.dot(s, a2);

        const t1 = f32(f32(sa1 - f32(a12 * sa2)) * invDet);
        const t2 = f32(f32(f32(a12 * sa1) - sa2) * invDet);

        const p1 = vec3.scale(t1, a1);
        const p2 = vec3.mulAdd(s, t2, a2);

        const g = vec3.sub(p2, p1);
        const g2 = vec3.lengthSq(g);
        if (g2 > f32(r * r)) {
            // Early out: closest point on infinite ray is outside infinite cylinder.
            return output;
        }

        const h = f32(Math.sqrt(f32(f32(f32(r * r) - g2) * invDet)));
        tr = f32(t2 - h);
    }

    // Outside ray?
    if (tr < 0 || f32(input.maxFraction * rayLength) < tr) {
        return output;
    }

    // The corresponding distance on the capsule axis. Length units.
    const tc = f32(u + f32(tr * a12));

    // Outside c1 end?
    if (tc < 0) {
        return rayCastSphere({ center: c1, radius: r }, input);
    }

    // Outside c2 end?
    if (length < tc) {
        return rayCastSphere({ center: c2, radius: r }, input);
    }

    // Hit point on capsule side, relative to c1.
    const p = vec3.mulAdd(s, tr, rayAxis);

    // Hit normal.
    const normal = vec3.normalize(vec3.mulSub(p, tc, axis));

    output.point = vec3.add(c1, p);
    output.normal = normal;
    output.fraction = clampf(f32(tr / rayLength), 0, input.maxFraction);
    output.hit = true;
    return output;
}

/** Shape cast against a capsule (b3ShapeCastCapsule). */
export function shapeCastCapsule(capsule: Capsule, input: ShapeCastInput): CastOutput {
    const pairInput: ShapeCastPairInput = {
        proxyA: { points: [capsule.center1, capsule.center2], count: 2, radius: capsule.radius },
        proxyB: input.proxy,
        transform: xf.identity(),
        translationB: input.translation,
        maxFraction: input.maxFraction,
        canEncroach: input.canEncroach,
    };
    return shapeCast(pairInput);
}

/**
 * Collision plane between a capsule mover and a sphere (b3CollideMoverAndSphere), both in the sphere's
 * frame. The normal points from the sphere toward the mover; a deep overlap (mover axis through the
 * center) falls back to a perpendicular of the mover axis so the normal is never degenerate.
 * @returns the plane, or null when they are separated.
 */
export function collideMoverAndSphere(shape: Sphere, mover: Capsule): PlaneResult | null {
    const totalRadius = f32(mover.radius + shape.radius);
    const closest = pointToSegmentDistance(mover.center1, mover.center2, shape.center);

    // The normal points from the sphere toward the mover.
    let { v: normal, length: distance } = getLengthAndNormalize(vec3.sub(closest, shape.center));

    if (distance > totalRadius) {
        return null;
    }

    const linearSlop = LINEAR_SLOP;
    if (distance < linearSlop) {
        // Deep overlap: the mover axis passes through the sphere center, so no direction is
        // preferred. Push perpendicular to the mover axis.
        const { v: axis, length } = getLengthAndNormalize(vec3.sub(mover.center2, mover.center1));
        normal = length > linearSlop ? vec3.perp(axis) : vec3.axisY();
        distance = 0;
    }

    const plane: Plane = { normal, offset: f32(totalRadius - distance) };
    return { plane, point: shape.center };
}

/**
 * Collision plane between a capsule mover and a capsule shape (b3CollideMoverAndCapsule), both in the
 * shape's frame. Normal points from the shape toward the mover; crossing/parallel core segments fall
 * back to a perpendicular of the mover axis.
 * @returns the plane, or null when they are separated.
 */
export function collideMoverAndCapsule(shape: Capsule, mover: Capsule): PlaneResult | null {
    const totalRadius = f32(mover.radius + shape.radius);

    const approach = segmentDistance(shape.center1, shape.center2, mover.center1, mover.center2);

    // The normal points from the shape toward the mover.
    let { v: normal, length: distance } = getLengthAndNormalize(
        vec3.sub(approach.point2, approach.point1),
    );

    if (distance > totalRadius) {
        return null;
    }

    const linearSlop = LINEAR_SLOP;
    if (distance < linearSlop) {
        // Deep overlap: the core segments intersect. Pick an arbitrary direction perpendicular to
        // the capsule axis.
        const { v: moverAxis, length: moverLength } = getLengthAndNormalize(
            vec3.sub(mover.center2, mover.center1),
        );
        normal = moverLength > linearSlop ? vec3.perp(moverAxis) : vec3.axisY();
        distance = 0;
    }

    const plane: Plane = { normal, offset: f32(totalRadius - distance) };
    return { plane, point: approach.point1 };
}
