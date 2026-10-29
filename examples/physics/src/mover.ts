import { Body, BodyType, GlobalTransform, type World } from "@dylanebert/shallot";
import {
    applyLinearImpulse,
    type CollisionPlane,
    castMover,
    castRayClosest,
    clipVector,
    collideMover,
    emptyQueryHit,
    getAngularVelocity,
    getInverseMass,
    getMassData,
    getWorldInverseRotationalInertia,
    solvePlanes,
} from "@dylanebert/shallot/standard/physics";

export type Vec = { x: number; y: number; z: number };
const f = Math.fround;
export const v = (x = 0, y = 0, z = 0): Vec => ({ x, y, z });
export const add = (a: Vec, b: Vec) => v(f(a.x + b.x), f(a.y + b.y), f(a.z + b.z));
export const sub = (a: Vec, b: Vec) => v(f(a.x - b.x), f(a.y - b.y), f(a.z - b.z));
export const mul = (s: number, a: Vec) => v(f(s * a.x), f(s * a.y), f(s * a.z));
export const dot = (a: Vec, b: Vec) => f(f(f(a.x * b.x) + f(a.y * b.y)) + f(a.z * b.z));
export const cross = (a: Vec, b: Vec) =>
    v(
        f(f(a.y * b.z) - f(a.z * b.y)),
        f(f(a.z * b.x) - f(a.x * b.z)),
        f(f(a.x * b.y) - f(a.y * b.x)),
    );
export const length = (a: Vec) => f(Math.sqrt(dot(a, a)));
const norm = (a: Vec) => {
    const n = length(a);
    return n < f(1.1920928955078125e-7) ? v() : mul(f(1 / n), a);
};
export function pose(world: World, body: number) {
    const g = world.storage(GlobalTransform);
    return {
        p: v(g.translation.x.get(body), g.translation.y.get(body), g.translation.z.get(body)),
        q: {
            v: v(g.rotation.x.get(body), g.rotation.y.get(body), g.rotation.z.get(body)),
            s: g.rotation.w.get(body),
        },
    };
}
export function rotate(q: { v: Vec; s: number }, a: Vec): Vec {
    const t = mul(2, cross(q.v, a));
    return add(add(a, mul(q.s, t)), cross(q.v, t));
}
export const moverState = () => ({
    position: v(7.5, 0.75, 9),
    velocity: v(),
    pogoVelocity: 0,
    pogoRay: undefined as { origin: Vec; end: Vec } | undefined,
    onGround: false,
    sprint: false,
    capsule: { center1: v(0, -0.5, 0), center2: v(0, 0.5, 0), radius: f(0.3) },
    planes: [] as CollisionPlane[],
    extras: [] as { body: number; shape: number; point: Vec }[],
    totalIterations: 0,
});
export type Mover = ReturnType<typeof moverState>;
export type MoverShape = { maxPush: number; clipVelocity: boolean; canMoverPush: boolean };

// samples/mover.cpp CharacterMover; input and the three shape policies belong to this scene.
export function move(
    world: World,
    s: Mover,
    forward: Vec,
    right: Vec,
    keys: Record<"W" | "A" | "S" | "D" | "Space" | "Shift", number>,
    clip: boolean,
    ignore: number,
    policies: Map<number, MoverShape>,
): void {
    const dt = f(1 / 60);
    forward = norm(v(forward.x, 0, forward.z));
    if (keys.Space && s.onGround) {
        s.velocity.y = 5;
        s.onGround = false;
    }
    s.sprint = s.onGround && !!keys.Shift;
    const speed = length(s.velocity);
    if (speed < f(0.01)) {
        s.velocity.x = 0;
        s.velocity.z = 0;
    } else {
        const control = speed < 1 ? 1 : speed;
        const drop = f(f(control * 4) * dt);
        const ratio = f(Math.max(0, f(speed - drop)) / speed);
        s.velocity.x = f(s.velocity.x * ratio);
        s.velocity.z = f(s.velocity.z * ratio);
    }
    const maxSpeed = s.sprint ? 9 : 6;
    const desiredVelocity = add(
        mul(f(maxSpeed * (keys.W - keys.S)), forward),
        mul(f(maxSpeed * (keys.D - keys.A)), right),
    );
    let desiredSpeed = length(desiredVelocity);
    const direction = norm(desiredVelocity);
    if (desiredSpeed > maxSpeed) desiredSpeed = maxSpeed;
    if (s.onGround) s.velocity.y = 0;
    const addSpeed = f(desiredSpeed - dot(s.velocity, direction));
    if (addSpeed > 0)
        s.velocity = add(s.velocity, mul(Math.min(f(f(30 * maxSpeed) * dt), addSpeed), direction));
    s.velocity.y = f(s.velocity.y - f(15 * dt));
    const rest = f(3 * s.capsule.radius),
        rayLength = f(rest + s.capsule.radius);
    const rayOrigin = add(s.position, s.capsule.center1);
    const ray = emptyQueryHit();
    const skipTeam = { categoryBits: 1n, maskBits: 0xfffffffdn };
    const hit = castRayClosest(world, rayOrigin, v(0, -rayLength, 0), ray, skipTeam);
    if (!hit || s.velocity.y > 0) {
        s.onGround = false;
        s.pogoVelocity = 0;
    } else {
        s.onGround = true;
        const currentLength = f(ray.fraction * rayLength);
        const omega = f(f(2 * f(Math.PI)) * 4),
            omegaH = f(omega * dt);
        s.pogoVelocity = f(
            f(s.pogoVelocity - f(f(omega * omegaH) * f(currentLength - rest))) /
                f(f(1 + f(f(2 * f(0.7)) * omegaH)) + f(omegaH * omegaH)),
        );
    }
    s.pogoRay = {
        origin: rayOrigin,
        end: s.onGround
            ? v(ray.point.x, ray.point.y, ray.point.z)
            : add(rayOrigin, v(0, -rayLength, 0)),
    };
    const start = s.position;
    const target = add(
        add(s.position, mul(dt, s.velocity)),
        mul(f(dt * s.pogoVelocity), v(0, 1, 0)),
    );
    s.totalIterations = 0;
    const accept = (shape: number) => shape !== ignore;
    for (let iteration = 0; iteration < 5; iteration++) {
        const hits = collideMover(world, s.position, s.capsule, 64, {
            categoryBits: 1n,
            maskBits: 0xffffffffn,
        });
        s.planes = [];
        s.extras = [];
        for (let i = 0; i < hits.count && s.planes.length < 8; i++) {
            const hit = hits.records[i]!;
            if (!accept(hit.shape)) continue;
            const policy = policies.get(hit.shape);
            s.planes.push({
                plane: structuredClone(hit.plane.plane),
                pushLimit: policy?.maxPush ?? f(3.4028234663852886e38),
                push: 0,
                clipVelocity: policy?.clipVelocity ?? true,
            });
            s.extras.push({
                body: hit.body,
                shape: hit.shape,
                point: add(s.position, hit.plane.point),
            });
        }
        const result = solvePlanes(sub(target, s.position), s.planes, s.planes.length);
        s.totalIterations += result.iterationCount;
        const { fraction } = castMover(
            world,
            s.position,
            s.capsule,
            result.delta,
            skipTeam,
            accept,
        );
        const delta = mul(fraction, result.delta);
        s.position = add(s.position, delta);
        if (dot(delta, delta) < f(f(0.01) * f(0.01))) break;
    }
    for (let i = 0; i < s.planes.length; i++) {
        const extra = s.extras[i]!;
        if (
            world.storage(Body).type.get(extra.body) !== BodyType.Dynamic ||
            policies.get(extra.shape)?.canMoverPush === false
        )
            continue;
        const mass = getMassData(world, extra.body)!;
        const placement = pose(world, extra.body);
        const center = add(placement.p, rotate(placement.q, mass.center));
        const normal = mul(-1, s.planes[i]!.plane.normal),
            r = sub(extra.point, center),
            rn = cross(r, normal);
        const inverse = getWorldInverseRotationalInertia(world, extra.body)!;
        const rotational = add(
            add(mul(rn.x, inverse.cx), mul(rn.y, inverse.cy)),
            mul(rn.z, inverse.cz),
        );
        const k = f(getInverseMass(world, extra.body)!.inverseMass + dot(rn, rotational));
        const normalMass = k > 0 ? f(1 / k) : 0;
        const g = world.storage(GlobalTransform).linearVelocity;
        const linear = v(g.x.get(extra.body), g.y.get(extra.body), g.z.get(extra.body));
        const angular = getAngularVelocity(world, extra.body)!;
        const vn = dot(sub(add(linear, cross(angular, r)), s.velocity), normal);
        const impulse = mul(Math.max(f(-normalMass * vn), 0), normal);
        s.velocity = sub(s.velocity, mul(0, impulse));
        applyLinearImpulse(world, extra.body, impulse, extra.point, true);
    }
    s.velocity = clip
        ? clipVector(s.velocity, s.planes, s.planes.length)
        : mul(f(1 / dt), sub(s.position, start));
}
