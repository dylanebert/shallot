// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

import {
    Body,
    BodyMotionLock,
    BodyType,
    type EntityRef,
    GlobalTransform,
    MoverJoint,
    PogoJoint,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    applyLinearImpulseToCenter,
    castRay,
    computeCosSin,
    getAngularVelocity,
    getLocalPoint,
    getMassData,
    getPogoJointCurrentSpeed,
    getPogoJointImpulse,
    PhysicsWorldDefinition,
} from "@dylanebert/shallot/standard/physics";
import { add, cross, dot, length, mul, pose, rotate, sub, type Vec, v } from "./mover";

const f = Math.fround;
const tuple = (p: Vec): [number, number, number] => [p.x, p.y, p.z];
const normalize = (p: Vec) => {
    return dot(p, p) > f(1000 * 1.1754943508222875e-38) ? mul(f(1 / length(p)), p) : v();
};
const velocity = (world: World, eid: number) => {
    const s = world.storage(GlobalTransform).linearVelocity;
    return v(s.x.get(eid), s.y.get(eid), s.z.get(eid));
};
const miss = () => ({
    point: v(),
    normal: v(),
    body: 0 as EntityRef,
    userMaterialId: 0n,
    fraction: 0,
    hit: false,
});

// samples/dynamic_mover.h and .cpp. Vendored gameplay, not an engine character policy.
export class DynamicMover {
    jumpSpeed = 7;
    maxSpeed = 6;
    minSpeed = f(0.1);
    stopSpeed = 3;
    accelerate = 20;
    airSteer = f(0.25);
    friction = 8;
    gravityScale = f(1.5);
    maxGroundForce = 6000;
    maxAirForce = 1500;
    pogoRestLength = f(0.9);
    pogoHertz = 3;
    pogoDampingRatio = f(0.8);
    pogoCompressionScale = 100;
    pogoTensionScale = 100;
    enableSkirtPogo = true;
    platformMaterialId = 7n;
    minGroundNormalY = f(0.7);
    capsule = { center1: v(0, -0.5, 0), center2: v(0, 0.5, 0), radius: f(0.3) };
    filter = { categoryBits: 2n, maskBits: 0xffffffffffffffbfn };
    ground = 0;
    body = 0;
    shape = 0;
    surfaceBody = 0 as EntityRef;
    moverJoint = 0;
    pogoJoint = 0;
    castResult = miss();
    velocity = v();
    timeSliding = 0;
    slideDuration = f(0.2);
    onGround = false;
    walkable = false;
    jumping = false;
    jumpTicks = 0;
    pogoImpulse = 0;
    pogoVelocity = 0;
    pogoLength = 0;
    probes: { origin: Vec; translation: Vec; fraction: number; hit: boolean }[] = [];
    selectedProbe = 0;

    create(world: World, position = v(0, 1.5, 0)) {
        this.velocity = v();
        this.onGround = this.walkable = this.jumping = false;
        this.jumpTicks =
            this.pogoImpulse =
            this.pogoVelocity =
            this.pogoLength =
            this.pogoJoint =
                0;
        this.ground = world.create();
        world.add(this.ground, Body);
        this.body = world.create();
        world.add(this.body, Transform, { translation: tuple(position) });
        world.add(this.body, Body, {
            type: BodyType.Dynamic,
            gravityScale: this.gravityScale,
            motionLocks:
                BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
            enableSleep: 0,
        });
        this.shape = world.create();
        world.add(this.shape, Shape, {
            body: this.body,
            kind: ShapeKind.Capsule,
            radius: this.capsule.radius,
            capsuleCenter1: tuple(this.capsule.center1),
            capsuleCenter2: tuple(this.capsule.center2),
            density: 200,
            friction: 0,
            restitution: 0,
            filterCategoryLow: Number(this.filter.categoryBits & 0xffffffffn),
            filterCategoryHigh: Number(this.filter.categoryBits >> 32n),
            filterMaskLow: Number(this.filter.maskBits & 0xffffffffn),
            filterMaskHigh: Number(this.filter.maskBits >> 32n),
            enableSensorEvents: 1,
        });
        this.createMoverJoint(world, this.ground);
    }
    private createMoverJoint(world: World, base: number) {
        this.moverJoint = world.create();
        world.add(this.moverJoint, MoverJoint, {
            a: base,
            b: this.body,
            linearVelocity: tuple(this.velocity),
            maxForce: this.maxGroundForce,
            collideConnected: 1,
        });
    }
    destroy(world: World) {
        for (const eid of [this.shape, this.body, this.ground, this.moverJoint, this.pogoJoint])
            if (eid) world.destroy(eid);
        this.ground = this.body = this.moverJoint = this.pogoJoint = 0;
    }
    setGravityScale(world: World, value: number) {
        this.gravityScale = f(value);
        if (this.body) world.storage(Body).gravityScale.set(this.body, value);
    }
    jump(world: World) {
        if (!this.onGround || !this.walkable) return false;
        let surfaceVelocity = 0;
        const surface = world.resolve(this.castResult.body);
        if (surface && world.has(surface, Body)) {
            const body = surface;
            const p = pose(world, body);
            const center = add(p.p, rotate(p.q, getMassData(world, body)!.center));
            surfaceVelocity = add(
                velocity(world, body),
                cross(getAngularVelocity(world, body)!, sub(this.castResult.point, center)),
            ).y;
        }
        const mass = getMassData(world, this.body)!.mass;
        const dv = f(
            Math.max(0, f(this.jumpSpeed - velocity(world, this.body).y)) + surfaceVelocity,
        );
        applyLinearImpulseToCenter(world, this.body, v(0, f(mass * dv), 0), true);
        this.launch();
        return true;
    }
    launch() {
        this.onGround = this.walkable = false;
        this.jumping = true;
        this.jumpTicks = 0;
    }
    update(
        world: World,
        dt: number,
        forward: Vec,
        right: Vec,
        throttle: { x: number; y: number },
        speedScale: number,
    ) {
        const rayLength = this.onGround ? f(2 * this.pogoRestLength) : this.pogoRestLength;
        const translation = v(0, -rayLength, 0);
        const origin = add(pose(world, this.body).p, this.capsule.center1);
        const cast = (origin: Vec) => {
            const result = miss();
            castRay(
                world,
                origin,
                translation,
                (hit) => {
                    if (hit.body === this.body || world.storage(Shape).isSensor.get(hit.shape))
                        return -1;
                    // Native samples use single-precision positions; public ray hits retain double precision.
                    Object.assign(result, {
                        point: v(f(hit.point.x), f(hit.point.y), f(hit.point.z)),
                        normal: { ...hit.normal },
                        body: world.ref(hit.body),
                        userMaterialId: hit.userMaterialId,
                        fraction: hit.fraction,
                        hit: true,
                    });
                    return hit.fraction;
                },
                this.filter,
            );
            this.probes.push({
                origin,
                translation,
                fraction: result.hit ? result.fraction : 1,
                hit: result.hit,
            });
            return result;
        };
        this.probes.length = 0;
        this.castResult = cast(origin);
        this.selectedProbe = 0;
        let pogoOffset = v();
        if (
            !(this.castResult.hit && this.castResult.normal.y >= this.minGroundNormalY) &&
            this.enableSkirtPogo
        ) {
            const vel = velocity(world, this.body);
            let heading = v(vel.x, 0, vel.z);
            if (dot(heading, heading) < 1.1920928955078125e-7) heading = v(1, 0, 0);
            heading = normalize(heading);
            const side = v(-heading.z, 0, heading.x);
            let best = 3.4028234663852886e38;
            for (let i = 0; i < 8; i++) {
                const cs = computeCosSin(world, f(f(f(2 * f(Math.PI)) * i) / 8));
                const offset = mul(this.capsule.radius, add(mul(cs[0], heading), mul(cs[1], side)));
                const result = cast(add(origin, offset));
                if (result.hit && result.normal.y >= this.minGroundNormalY) {
                    const displacement = Math.abs(
                        f(f(result.fraction * rayLength) - this.pogoRestLength),
                    );
                    if (displacement < best) {
                        best = displacement;
                        this.castResult = result;
                        this.selectedProbe = this.probes.length - 1;
                        pogoOffset = offset;
                    }
                }
            }
        }
        const hit = this.castResult;
        const hitBody = world.resolve(hit.body);
        if (this.jumping) {
            this.jumpTicks++;
            if (this.jumpTicks > 2 && hit.hit && dot(this.velocity, hit.normal) <= 0)
                this.jumping = false;
        }
        if (hit.hit) {
            this.walkable = hit.normal.y >= this.minGroundNormalY;
            this.onGround = !this.jumping;
            if (this.onGround)
                this.timeSliding = this.walkable
                    ? Math.max(f(this.timeSliding - dt), 0)
                    : Math.min(f(this.timeSliding + dt), this.slideDuration);
        } else {
            this.onGround = this.walkable = false;
            this.timeSliding = Math.max(f(this.timeSliding - dt), 0);
        }
        let base = this.ground;
        if (
            hit.hit &&
            this.platformMaterialId !== 0n &&
            hit.userMaterialId === this.platformMaterialId &&
            world.storage(Body).type.get(hitBody) === BodyType.Kinematic
        )
            base = hitBody;
        if (
            !world.has(this.moverJoint, MoverJoint) ||
            world.storage(MoverJoint).a.get(this.moverJoint) !== base
        ) {
            if (world.has(this.moverJoint, MoverJoint)) world.destroy(this.moverJoint);
            this.createMoverJoint(world, base);
        }
        this.velocity = sub(velocity(world, this.body), velocity(world, base));
        if (this.onGround) {
            const speed = f(
                Math.sqrt(
                    f(f(this.velocity.x * this.velocity.x) + f(this.velocity.z * this.velocity.z)),
                ),
            );
            if (speed < this.minSpeed) this.velocity.x = this.velocity.z = 0;
            else {
                const drop = f(f(Math.max(speed, this.stopSpeed) * this.friction) * dt);
                const ratio = f(Math.max(0, f(speed - drop)) / speed);
                this.velocity.x = f(this.velocity.x * ratio);
                this.velocity.z = f(this.velocity.z * ratio);
            }
        }
        const maxForce = this.onGround
            ? this.slideDuration > 0
                ? f(
                      Math.max(0, Math.min(1, f(1 - f(this.timeSliding / this.slideDuration)))) *
                          this.maxGroundForce,
                  )
                : this.walkable
                  ? this.maxGroundForce
                  : 0
            : this.maxAirForce;
        world.storage(MoverJoint).maxForce.set(this.moverJoint, maxForce);
        const maxSpeed = f(speedScale * this.maxSpeed);
        const desired = add(
            mul(f(maxSpeed * throttle.x), forward),
            mul(f(maxSpeed * throttle.y), right),
        );
        desired.y = 0;
        const direction = normalize(desired);
        const desiredSpeed = Math.min(
            dot(desired, desired) > f(1000 * 1.1754943508222875e-38) ? length(desired) : 0,
            maxSpeed,
        );
        const addSpeed = f(desiredSpeed - dot(this.velocity, direction));
        if (addSpeed > 0) {
            const steer = this.walkable ? 1 : this.airSteer;
            const accel = Math.min(f(f(f(steer * this.accelerate) * maxSpeed) * dt), addSpeed);
            this.velocity.x = f(this.velocity.x + f(accel * direction.x));
            this.velocity.z = f(this.velocity.z + f(accel * direction.z));
        }
        if (
            hit.hit &&
            Math.abs(f(dot(hit.normal, hit.normal) - 1)) < f(100 * 1.1920928955078125e-7)
        ) {
            this.pogoLength = f(hit.fraction * rayLength);
            const gravity = world.resource(PhysicsWorldDefinition).gravity;
            const weight = f(
                f(this.gravityScale * length(gravity)) * getMassData(world, this.body)!.mass,
            );
            const anchorA = getLocalPoint(world, hitBody, hit.point)!;
            const anchorB = add(this.capsule.center1, pogoOffset);
            const tension = this.jumping ? 0 : f(this.pogoTensionScale * weight);
            const compression = f(this.pogoCompressionScale * weight);
            const impulse =
                this.pogoJoint && world.has(this.pogoJoint, PogoJoint)
                    ? getPogoJointImpulse(world, this.pogoJoint)
                    : null;
            const valid = impulse !== null;
            if (hit.body !== this.surfaceBody || !valid) {
                if (valid) {
                    this.pogoImpulse = impulse!;
                    this.pogoVelocity = getPogoJointCurrentSpeed(world, this.pogoJoint)!;
                    world.destroy(this.pogoJoint);
                }
                this.surfaceBody = hit.body;
                this.pogoJoint = world.create();
                world.add(this.pogoJoint, PogoJoint, {
                    a: hitBody,
                    b: this.body,
                    collideConnected: 1,
                    localAnchorA: tuple(anchorA),
                    localAnchorB: tuple(anchorB),
                    normal: tuple(hit.normal),
                    restLength: this.pogoRestLength,
                    hertz: this.pogoHertz,
                    dampingRatio: this.pogoDampingRatio,
                    maxCompressionForce: compression,
                    maxTensionForce: tension,
                    impulse: this.pogoImpulse,
                    velocity: this.pogoVelocity,
                });
            } else {
                const p = world.storage(PogoJoint);
                p.localAnchorA.set(this.pogoJoint, ...tuple(anchorA));
                p.localAnchorB.set(this.pogoJoint, ...tuple(anchorB));
                p.normal.set(this.pogoJoint, ...tuple(hit.normal));
                p.maxTensionForce.set(this.pogoJoint, tension);
                p.maxCompressionForce.set(this.pogoJoint, compression);
                p.hertz.set(this.pogoJoint, this.pogoHertz);
                p.dampingRatio.set(this.pogoJoint, this.pogoDampingRatio);
            }
        } else {
            if (this.pogoJoint && world.has(this.pogoJoint, PogoJoint))
                world.destroy(this.pogoJoint);
            this.pogoJoint = 0;
            this.surfaceBody = 0 as EntityRef;
            this.pogoImpulse = this.pogoVelocity = this.pogoLength = 0;
        }
        world.storage(MoverJoint).linearVelocity.set(this.moverJoint, ...tuple(this.velocity));
    }
}
