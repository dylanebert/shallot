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
export type MoverShape = { maxPush: number; clipVelocity: boolean; canMoverPush: boolean };
export type Mover = GeometricMover;

// samples/geometric_mover.cpp; shape policies use ECS entities instead of shape user-data pointers.
export class GeometricMover {
    jumpSpeed = 5;
    maxSpeed = 6;
    minSpeed = f(0.01);
    stopSpeed = 1;
    accelerate = 30;
    friction = 4;
    gravity = 15;
    pogoRestLength = f(0.9);
    pogoHertz = 4;
    pogoDampingRatio = f(0.7);
    position = v(7.5, 0.75, 9);
    velocity = v();
    pogoVelocity = 0;
    pogoRay: { origin: Vec; end: Vec } | undefined;
    onGround = false;
    sprint = false;
    capsule = { center1: v(0, -0.5, 0), center2: v(0, 0.5, 0), radius: f(0.3) };
    planes: CollisionPlane[] = [];
    extras: { body: number; shape: number; point: Vec }[] = [];
    totalIterations = 0;

    jump() {
        if (!this.onGround) return false;
        this.velocity.y = this.jumpSpeed;
        this.onGround = false;
        return true;
    }
    update(
        world: World,
        dt: number,
        forward: Vec,
        right: Vec,
        throttle: { x: number; y: number },
        speedScale: number,
        clip: boolean,
        ignore: number,
        policies: Map<number, MoverShape>,
    ): void {
        forward = norm(v(forward.x, 0, forward.z));
        const speed = length(this.velocity);
        if (speed < this.minSpeed) {
            this.velocity.x = 0;
            this.velocity.z = 0;
        } else {
            const control = speed < this.stopSpeed ? this.stopSpeed : speed;
            const drop = f(f(control * this.friction) * dt);
            const ratio = f(Math.max(0, f(speed - drop)) / speed);
            this.velocity.x = f(this.velocity.x * ratio);
            this.velocity.z = f(this.velocity.z * ratio);
        }
        const maxSpeed = f(speedScale * this.maxSpeed);
        const desiredVelocity = add(
            mul(f(maxSpeed * throttle.x), forward),
            mul(f(maxSpeed * throttle.y), right),
        );
        let desiredSpeed = length(desiredVelocity);
        const direction = norm(desiredVelocity);
        if (desiredSpeed > maxSpeed) desiredSpeed = maxSpeed;
        if (this.onGround) this.velocity.y = 0;
        const addSpeed = f(desiredSpeed - dot(this.velocity, direction));
        if (addSpeed > 0)
            this.velocity = add(
                this.velocity,
                mul(Math.min(f(f(this.accelerate * maxSpeed) * dt), addSpeed), direction),
            );
        this.velocity.y = f(this.velocity.y - f(this.gravity * dt));
        const rest = this.pogoRestLength,
            rayLength = f(rest + this.capsule.radius);
        const rayOrigin = add(this.position, this.capsule.center1);
        const ray = emptyQueryHit();
        const skipTeam = { categoryBits: 1n, maskBits: 0xfffffffffffffffdn };
        const hit = castRayClosest(world, rayOrigin, v(0, -rayLength, 0), ray, skipTeam);
        if (!hit || this.velocity.y > 0) {
            this.onGround = false;
            this.pogoVelocity = 0;
        } else {
            this.onGround = true;
            const currentLength = f(ray.fraction * rayLength);
            const omega = f(f(2 * f(Math.PI)) * this.pogoHertz),
                omegaH = f(omega * dt);
            this.pogoVelocity = f(
                f(this.pogoVelocity - f(f(omega * omegaH) * f(currentLength - rest))) /
                    f(f(1 + f(f(2 * this.pogoDampingRatio) * omegaH)) + f(omegaH * omegaH)),
            );
        }
        this.pogoRay = {
            origin: rayOrigin,
            end: this.onGround
                ? v(ray.point.x, ray.point.y, ray.point.z)
                : add(rayOrigin, v(0, -rayLength, 0)),
        };
        const start = this.position;
        const target = add(
            add(this.position, mul(dt, this.velocity)),
            mul(f(dt * this.pogoVelocity), v(0, 1, 0)),
        );
        this.totalIterations = 0;
        const accept = (shape: number) => shape !== ignore;
        for (let iteration = 0; iteration < 5; iteration++) {
            const hits = collideMover(world, this.position, this.capsule, 64, {
                categoryBits: 1n,
                maskBits: 0xffffffffffffffffn,
            });
            this.planes = [];
            this.extras = [];
            for (let i = 0; i < hits.count && this.planes.length < 8; i++) {
                const hit = hits.records[i]!;
                if (!accept(hit.shape)) continue;
                const policy = policies.get(hit.shape);
                this.planes.push({
                    plane: structuredClone(hit.plane.plane),
                    pushLimit: policy?.maxPush ?? f(3.4028234663852886e38),
                    push: 0,
                    clipVelocity: policy?.clipVelocity ?? true,
                });
                this.extras.push({
                    body: hit.body,
                    shape: hit.shape,
                    point: add(this.position, hit.plane.point),
                });
            }
            const result = solvePlanes(sub(target, this.position), this.planes, this.planes.length);
            this.totalIterations += result.iterationCount;
            const { fraction } = castMover(
                world,
                this.position,
                this.capsule,
                result.delta,
                skipTeam,
                accept,
            );
            const delta = mul(fraction, result.delta);
            this.position = add(this.position, delta);
            if (dot(delta, delta) < f(f(0.01) * f(0.01))) break;
        }
        this.pushBodies(world, policies);
        this.velocity = clip
            ? clipVector(this.velocity, this.planes, this.planes.length)
            : mul(f(1 / dt), sub(this.position, start));
    }
    private pushBodies(world: World, policies: Map<number, MoverShape>) {
        for (let i = 0; i < this.planes.length; i++) {
            const extra = this.extras[i]!;
            if (
                world.storage(Body).type.get(extra.body) !== BodyType.Dynamic ||
                policies.get(extra.shape)?.canMoverPush === false
            )
                continue;
            const mass = getMassData(world, extra.body)!;
            const placement = pose(world, extra.body);
            const center = add(placement.p, rotate(placement.q, mass.center));
            const normal = mul(-1, this.planes[i]!.plane.normal),
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
            const vn = dot(sub(add(linear, cross(angular, r)), this.velocity), normal);
            const impulse = mul(Math.max(f(-normalMass * vn), 0), normal);
            this.velocity = sub(this.velocity, mul(0, impulse));
            applyLinearImpulse(world, extra.body, impulse, extra.point, true);
        }
    }
}
