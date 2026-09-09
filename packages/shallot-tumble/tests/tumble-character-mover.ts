// The tumble.js `Character` sample (`samples/src/samples/character.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A kinematic capsule mover patrols a walled arena, climbing a ramp and steps on
// its pogo ground-follow and shoving loose crates — self-driven, no input: the sample's `update()` advances
// a heading in a slow arc every step and always drives the mover forward, so the trajectory is fully
// deterministic and gold-exact.
//
// The mover is NOT a world body — it lives only as a plain-object query volume. Its per-step `solve()`
// (collide → solvePlanes → castMover sweep + a pogo ray, then a push impulse on dynamic bodies it leans on)
// DOES mutate the world through `push()`, so it is load-bearing for the gold and ports whole. The mover's own
// wireframe capsule + grounded/speed readout are the sample's `render()` overlay ({@link renderCharacterMover}),
// restored here (green grounded / orange airborne) so the invisible mover is visible again.
//
// Creation order is load-bearing for the hash: floor, four walls, ramp, three steps, four crates — the
// sample's exact order. The mover is constructed after, creating no body.

import {
    BodyType,
    type Capsule,
    type CollisionPlane,
    clipVector,
    makeBoxHull,
    type Shape,
    solvePlanes,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams, SampleUpdate } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

const IDENT_Y = { x: 0, y: 1, z: 0 };
const FLT_MAX = 3.4028235e38;

// Tuning from Box3D's CharacterMover (samples/sample.h). Units: m, s.
const MAX_SPEED = 6;
const MIN_SPEED = 0.01;
const STOP_SPEED = 1;
const ACCELERATE = 30;
const FRICTION = 4;
const GRAVITY = 15;

// Sample-local vector math on `{ x, y, z }` (the sample's `gfx/mat4.ts` `vec`): plain JS `Math`, so a
// faithful port is bit-identical to the mint by construction (the compound-trig authoring precedent).
const vec = {
    sub: (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
    add: (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
    scale: (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s }),
    dot: (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z,
    length: (a: Vec3): number => Math.hypot(a.x, a.y, a.z),
    cross: (a: Vec3, b: Vec3): Vec3 => ({
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }),
    mulAdd: (a: Vec3, s: number, b: Vec3): Vec3 => ({
        x: a.x + s * b.x,
        y: a.y + s * b.y,
        z: a.z + s * b.z,
    }),
};

type Extra = { point: Vec3; shape: Shape };

// The mover state and its per-step solve. A plain object updated in place — the capsule never becomes a
// world body; it exists only as a query volume.
class Mover {
    pos: Vec3;
    velocity: Vec3 = { x: 0, y: 0, z: 0 };
    readonly capsule: Capsule = {
        center1: { x: 0, y: -0.5, z: 0 },
        center2: { x: 0, y: 0.5, z: 0 },
        radius: 0.3,
    };
    onGround = false;
    private _pogoVelocity = 0;
    private _planes: CollisionPlane[] = [];
    private _extras: Extra[] = [];

    constructor(
        private readonly _world: World,
        position: Vec3,
    ) {
        this.pos = { ...position };
    }

    // Resolve one step of motion for a throttle in the mover's local frame (x = forward, y = right).
    solve(dt: number, forward: Vec3, right: Vec3, throttle: { x: number; y: number }): void {
        // Ground friction: kill tiny speeds, otherwise shed speed at a fixed rate.
        const speed = vec.length({ x: this.velocity.x, y: 0, z: this.velocity.z });
        if (speed < MIN_SPEED) {
            this.velocity.x = 0;
            this.velocity.z = 0;
        } else {
            const control = speed < STOP_SPEED ? STOP_SPEED : speed;
            const drop = control * FRICTION * dt;
            const newSpeed = Math.max(0, speed - drop) / speed;
            this.velocity.x *= newSpeed;
            this.velocity.z *= newSpeed;
        }

        // Accelerate toward the desired velocity (quake-style: only add the missing component).
        const desired = vec.add(
            vec.scale(forward, MAX_SPEED * throttle.x),
            vec.scale(right, MAX_SPEED * throttle.y),
        );
        let desiredSpeed = vec.length(desired);
        const desiredDir =
            desiredSpeed > 0 ? vec.scale(desired, 1 / desiredSpeed) : { x: 0, y: 0, z: 0 };
        if (desiredSpeed > MAX_SPEED) desiredSpeed = MAX_SPEED;

        if (this.onGround) this.velocity.y = 0;

        const current = vec.dot(this.velocity, desiredDir);
        const add = desiredSpeed - current;
        if (add > 0) {
            const accel = Math.min(ACCELERATE * MAX_SPEED * dt, add);
            this.velocity = vec.mulAdd(this.velocity, accel, desiredDir);
        }

        this.velocity.y -= GRAVITY * dt;

        // Pogo ray: a critically-damped spring that keeps the capsule floating a fixed height over
        // whatever it's standing on, so it climbs steps and ramps without a rigid-body contact.
        const restLength = 3 * this.capsule.radius;
        const rayLength = restLength + this.capsule.radius;
        const rayOrigin = vec.add(this.pos, this.capsule.center1);
        const ray = this._world.castRayClosest(rayOrigin, { x: 0, y: -rayLength, z: 0 });
        if (ray.hit === false) {
            this.onGround = false;
            this._pogoVelocity = 0;
        } else {
            this.onGround = true;
            const length = ray.fraction * rayLength;
            const zeta = 0.7;
            const omega = 2 * Math.PI * 4;
            const omegaH = omega * dt;
            this._pogoVelocity =
                (this._pogoVelocity - omega * omegaH * (length - restLength)) /
                (1 + 2 * zeta * omegaH + omegaH * omegaH);
        }

        const target = vec.mulAdd(
            vec.mulAdd(this.pos, dt, this.velocity),
            dt * this._pogoVelocity,
            IDENT_Y,
        );

        // Iterate collide → solve → cast so the mover slides along surfaces instead of stopping dead.
        for (let it = 0; it < 5; ++it) {
            this.gather();
            const targetDelta = vec.sub(target, this.pos);
            const result = solvePlanes(targetDelta, this._planes, this._planes.length);
            const fraction = this._world.castMover(this.pos, this.capsule, result.delta);
            const delta = vec.scale(result.delta, fraction);
            this.pos = vec.add(this.pos, delta);
            if (vec.dot(delta, delta) < 0.0001) break;
        }

        this.push();
        this.velocity = clipVector(this.velocity, this._planes, this._planes.length);
    }

    // Collect the collision planes (and their contact points) around the mover's current position.
    private gather(): void {
        this._planes = [];
        this._extras = [];
        this._world.collideMover(this.pos, this.capsule, (shape, results) => {
            for (const pr of results) {
                if (this._planes.length >= 8) break;
                this._planes.push({
                    plane: pr.plane,
                    pushLimit: FLT_MAX,
                    push: 0,
                    clipVelocity: true,
                });
                this._extras.push({ point: vec.add(this.pos, pr.point), shape });
            }
            return true;
        });
    }

    // Push the dynamic bodies the mover is leaning on. A simplified normal impulse (inverse-mass only,
    // omitting the rotational term of Box3D's full solve) — enough to shove crates convincingly.
    private push(): void {
        for (let i = 0; i < this._planes.length; ++i) {
            const body = this._extras[i].shape.getBody();
            if (body.getType() !== BodyType.Dynamic) continue;
            const mass = body.getMassData().mass;
            if (mass <= 0) continue;

            const point = this._extras[i].point;
            const normal = vec.scale(this._planes[i].plane.normal, -1);
            const rB = vec.sub(point, body.getWorldCenterOfMass());
            const vB = vec.add(body.getLinearVelocity(), vec.cross(body.getAngularVelocity(), rB));
            const vn = vec.dot(vec.sub(vB, this.velocity), normal);
            const impulse = Math.max(-mass * vn, 0);
            if (impulse > 0) body.applyLinearImpulse(vec.scale(normal, impulse), point, true);
        }
    }
}

let mover: Mover | null = null;
let heading = 0;

/**
 * Author the Character scene into `world`: a floor, four perimeter walls, a tilted ramp, three steps, and
 * four loose crates the mover shoves. The kinematic capsule mover is constructed after (creating no body) and
 * driven in {@link updateCharacterMover}.
 */
export function buildCharacterMover(world: World, _params: SampleParams): void {
    heading = 0;

    // Floor + perimeter walls contain the patrol and the crates it shoves.
    world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(9, 0.5, 9));
    const wall = (x: number, z: number, hx: number, hz: number): void => {
        world
            .createBody({ type: BodyType.Static, position: { x, y: 1, z } })
            .createHull({}, makeBoxHull(hx, 1, hz));
    };
    wall(0, -9, 9, 0.3);
    wall(0, 9, 9, 0.3);
    wall(-9, 0, 0.3, 9);
    wall(9, 0, 0.3, 9);

    // A ramp and a pair of steps for the pogo ground-follow to climb.
    const tilt = (18 * Math.PI) / 180;
    world
        .createBody({
            type: BodyType.Static,
            position: { x: 5, y: 0.8, z: -4 },
            rotation: { v: { x: Math.sin(tilt / 2), y: 0, z: 0 }, s: Math.cos(tilt / 2) },
        })
        .createHull({}, makeBoxHull(2.5, 0.25, 3));
    for (let i = 0; i < 3; ++i) {
        world
            .createBody({
                type: BodyType.Static,
                position: { x: -5, y: 0.4 + 0.5 * i, z: 3 - i },
            })
            .createHull({}, makeBoxHull(2, 0.25, 1));
    }

    // Loose crates to bump around.
    const crate = makeBoxHull(0.5, 0.5, 0.5);
    for (let i = 0; i < 4; ++i) {
        world
            .createBody({ type: BodyType.Dynamic, position: { x: -1 + i, y: 1, z: 0 } })
            .createHull({ density: 0.5 }, crate);
    }

    mover = new Mover(world, { x: 0, y: 2, z: 6 });
}

/**
 * Steer the mover in a slow arc so it patrols the arena (the sample's `update()`); the plane solver handles
 * the obstacles. Self-driven — the heading advances each step and the throttle is always forward.
 */
export const updateCharacterMover: SampleUpdate = (
    _world: World,
    _params: SampleParams,
    dt: number,
) => {
    if (mover === null) return;
    heading += 0.9 * dt;
    const forward = { x: Math.cos(heading), y: 0, z: Math.sin(heading) };
    const right = { x: -Math.sin(heading), y: 0, z: Math.cos(heading) };
    mover.solve(dt, forward, right, { x: 1, y: 0 });
};

/**
 * Draw the mover's capsule (green grounded / orange airborne) and read out its ground state + speed (the
 * sample's `render()`). The mover is not a world body, so this is its only visual presence.
 */
export const renderCharacterMover: SampleRender = (draw: Overlay) => {
    if (mover === null) return;
    draw.solidCapsule(
        { p: mover.pos, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        mover.capsule,
        mover.onGround ? 0x40c040 : 0xc06040,
    );
    const speed = vec.length(mover.velocity);
    draw.text(`${mover.onGround ? "grounded" : "airborne"}   ${speed.toFixed(1)} m/s`);
};
