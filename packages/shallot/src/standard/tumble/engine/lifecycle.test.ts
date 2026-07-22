// Stage 7 (lifecycle) tests: world/body/shape create-destroy, generational validity, and the
// mass-accumulation path (b3UpdateBodyMassData wiring: sum shape masses, parallel-axis shift to the
// body center, invert). Ported from test_body.c (FarSingle/CubeSphereMass) and test_world.c
// (TestIsValid). The mass tolerances are analytic (the underlying computeSphereMass / steiner /
// invertT are already bit-exact gold-tested); per-step bit-exact fixtures arrive with the solver
// stage. The stepping subtests (HelloWorld, EmptyWorld, ...) belong to that stage.

import { describe, expect, test } from "bun:test";
import { computeSphereMass } from "./geometry";
import {
    BodyType,
    createCompound,
    createGrid,
    defaultSurfaceMaterial,
    type MassData,
    makeBoxHull,
    type Vec3,
    World,
} from "./index";
import { f32 } from "./math";

const PI = Math.PI;

function sphereBodyMass(centers: Vec3[], radius: number, density: number): MassData {
    const world = new World();
    const body = world.createBody({ type: BodyType.Dynamic });
    for (const center of centers) {
        body.createSphere({ density }, { center, radius });
    }
    body.applyMassFromShapes();
    const md = body.getMassData();
    world.destroy();
    return md;
}

describe("body mass accumulation", () => {
    // One sphere far from the body origin: the center of mass lands on the sphere and the inertia
    // about it is the bare central inertia, with no trace of the offset.
    test("far single sphere", () => {
        const radius = 0.5;
        const density = 1.0;
        const center: Vec3 = { x: 100, y: -50, z: 75 };
        const md = sphereBodyMass([center], radius, density);

        const mass = density * (4 / 3) * PI * radius * radius * radius;
        const central = 0.4 * mass * radius * radius;

        expect(Math.abs(md.mass - mass)).toBeLessThan(1e-4);
        expect(Math.abs(md.center.x - center.x)).toBeLessThan(1e-3);
        expect(Math.abs(md.center.y - center.y)).toBeLessThan(1e-3);
        expect(Math.abs(md.center.z - center.z)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cx.x - central)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cy.y - central)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cz.z - central)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cy.x)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cz.x)).toBeLessThan(1e-3);
        expect(Math.abs(md.inertia.cz.y)).toBeLessThan(1e-3);
    });

    // Eight equal spheres on the corners of a cube parked far from the origin. The center is the
    // cube center; products of inertia cancel by symmetry so the tensor stays diagonal.
    test("far cube of spheres", () => {
        const radius = 0.5;
        const density = 1.0;
        const h = 1.0;
        const p: Vec3 = { x: 100, y: 100, z: 100 };

        const centers: Vec3[] = [];
        for (let sx = -1; sx <= 1; sx += 2) {
            for (let sy = -1; sy <= 1; sy += 2) {
                for (let sz = -1; sz <= 1; sz += 2) {
                    centers.push({ x: p.x + sx * h, y: p.y + sy * h, z: p.z + sz * h });
                }
            }
        }

        const md = sphereBodyMass(centers, radius, density);

        const mass = density * (4 / 3) * PI * radius * radius * radius;
        const totalMass = 8 * mass;
        const diag = 8 * 0.4 * mass * radius * radius + 16 * mass * h * h;

        expect(Math.abs(md.mass - totalMass)).toBeLessThan(1e-3);
        expect(Math.abs(md.center.x - p.x)).toBeLessThan(1e-2);
        expect(Math.abs(md.center.y - p.y)).toBeLessThan(1e-2);
        expect(Math.abs(md.center.z - p.z)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cx.x - diag)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cy.y - diag)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cz.z - diag)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cy.x)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cz.x)).toBeLessThan(1e-2);
        expect(Math.abs(md.inertia.cz.y)).toBeLessThan(1e-2);
    });
});

describe("body velocity + center of mass", () => {
    const noLocks = {
        linearX: false,
        linearY: false,
        linearZ: false,
        angularX: false,
        angularY: false,
        angularZ: false,
    };

    test("velocity setters round-trip on a dynamic body", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic });
        body.setLinearVelocity({ x: 1, y: 2, z: 3 });
        body.setAngularVelocity({ x: 4, y: 5, z: 6 });
        expect(body.getLinearVelocity()).toEqual({ x: 1, y: 2, z: 3 });
        expect(body.getAngularVelocity()).toEqual({ x: 4, y: 5, z: 6 });
        world.destroy();
    });

    test("setLinearVelocity does not retain the caller's vector", () => {
        // Finalize writes the solved velocity into the body state in place; if the setter stored the
        // caller's object instead of copying it, stepping would clobber the user's vector.
        const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        const velocity = { x: 1, y: 2, z: 3 };
        body.setLinearVelocity(velocity);
        world.step(1 / 60, 4);
        expect(body.getLinearVelocity().y).toBeLessThan(2); // gravity acted — the body simulated
        expect(velocity).toEqual({ x: 1, y: 2, z: 3 });
        world.destroy();
    });

    test("setters are a no-op on a static body", () => {
        const world = new World();
        const body = world.createBody({});
        body.setLinearVelocity({ x: 1, y: 2, z: 3 });
        body.setAngularVelocity({ x: 4, y: 5, z: 6 });
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        expect(body.getAngularVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        world.destroy();
    });

    test("setAngularVelocity masks locked axes", () => {
        const world = new World();
        const body = world.createBody({
            type: BodyType.Dynamic,
            motionLocks: { ...noLocks, angularY: true },
        });
        body.setAngularVelocity({ x: 4, y: 5, z: 6 });
        expect(body.getAngularVelocity()).toEqual({ x: 4, y: 0, z: 6 });
        world.destroy();
    });

    test("getWorldCenterOfMass tracks the body position + shape", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 10, y: 0, z: -5 } });
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        body.applyMassFromShapes();
        const com = body.getWorldCenterOfMass();
        expect(Math.abs(com.x - 10)).toBeLessThan(1e-5);
        expect(Math.abs(com.y - 0)).toBeLessThan(1e-5);
        expect(Math.abs(com.z + 5)).toBeLessThan(1e-5);
        world.destroy();
    });
});

describe("setTargetTransform", () => {
    const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };

    test("translation sets linear velocity = (target - current) / dt", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Kinematic });
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        body.applyMassFromShapes();
        body.setTargetTransform({ p: { x: 2, y: 4, z: 6 }, q: identity }, 2.0);
        expect(body.getLinearVelocity()).toEqual({ x: 1, y: 2, z: 3 });
        expect(body.getAngularVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        world.destroy();
    });

    test("rotation sets angular velocity = 2*(q2-q1)*conj(q1)/dt", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Kinematic });
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));
        body.applyMassFromShapes();
        // A +90° rotation about z from identity: w = 2 * sin(45°) / dt about z.
        const s = Math.sin(Math.PI / 4);
        body.setTargetTransform(
            { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: s }, s } },
            1.0,
        );
        const w = body.getAngularVelocity();
        expect(Math.abs(w.x)).toBeLessThan(1e-6);
        expect(Math.abs(w.y)).toBeLessThan(1e-6);
        expect(Math.abs(w.z - 2 * s)).toBeLessThan(1e-5);
        world.destroy();
    });

    test("no-op on a static body", () => {
        const world = new World();
        const body = world.createBody({});
        body.setTargetTransform({ p: { x: 1, y: 1, z: 1 }, q: identity }, 1.0);
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        world.destroy();
    });

    test("no-op when timeStep <= 0", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Kinematic });
        body.setTargetTransform({ p: { x: 1, y: 1, z: 1 }, q: identity }, 0);
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        world.destroy();
    });

    test("no-op on a disabled body", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Kinematic, isEnabled: false });
        body.setTargetTransform({ p: { x: 1, y: 1, z: 1 }, q: identity }, 1.0);
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        world.destroy();
    });
});

describe("generational validity", () => {
    // Bodies and worlds invalidate their handles on destroy; a destroyed world invalidates every
    // handle into it (test_world.c TestIsValid).
    test("body and world validity across destroy", () => {
        const world = new World();
        expect(world.isValid()).toBe(true);

        const body1 = world.createBody({});
        expect(body1.isValid()).toBe(true);
        const body2 = world.createBody({});
        expect(body2.isValid()).toBe(true);

        body1.destroy();
        expect(body1.isValid()).toBe(false);
        expect(body2.isValid()).toBe(true);

        body2.destroy();
        expect(body2.isValid()).toBe(false);

        world.destroy();
        expect(world.isValid()).toBe(false);
        expect(body1.isValid()).toBe(false);
        expect(body2.isValid()).toBe(false);
    });

    // A recycled body slot gets a fresh generation, so the prior handle stays invalid.
    test("recycled body slot invalidates the old handle", () => {
        const world = new World();
        const b1 = world.createBody({ type: BodyType.Dynamic });
        b1.destroy();
        const b2 = world.createBody({ type: BodyType.Dynamic });
        expect(b2.isValid()).toBe(true);
        expect(b1.isValid()).toBe(false); // same slot, bumped generation
        world.destroy();
    });

    // Shapes follow the same idiom as bodies but key on their own id sentinel: destroy invalidates
    // the handle, a recycled slot bumps the generation, and a dead world invalidates every shape.
    test("shape validity across destroy, recycle, and world teardown", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic });
        const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 0.5 };
        const s1 = body.createSphere({ density: 1 }, sphere);
        expect(s1.isValid()).toBe(true);

        s1.destroy();
        expect(s1.isValid()).toBe(false);

        const s2 = body.createSphere({ density: 1 }, sphere);
        expect(s2.isValid()).toBe(true);
        expect(s1.isValid()).toBe(false); // recycled slot, bumped generation

        world.destroy();
        expect(s2.isValid()).toBe(false);
    });
});

describe("lifecycle bookkeeping", () => {
    test("counters track create and destroy", () => {
        const world = new World();
        let c = world.getCounters();
        expect(c.bodyCount).toBe(0);
        expect(c.shapeCount).toBe(0);

        const ground = world.createBody({});
        ground.createHull({}, makeBoxHull(50, 10, 50));

        const box = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        box.createHull({ density: 1 }, makeBoxHull(1, 1, 1));

        c = world.getCounters();
        expect(c.bodyCount).toBe(2);
        expect(c.shapeCount).toBe(2);
        expect(box.getShapeCount()).toBe(1);
        expect(box.getMass()).toBeGreaterThan(0);

        box.destroy();
        c = world.getCounters();
        expect(c.bodyCount).toBe(1);
        expect(c.shapeCount).toBe(1);

        world.destroy();
    });

    test("dynamic hull body gets an island; static body does not", () => {
        const world = new World();
        // A dynamic body lands in the awake set with an island → islandCount 1.
        const dyn = world.createBody({ type: BodyType.Dynamic });
        dyn.createHull({ density: 1 }, makeBoxHull(1, 1, 1));
        expect(world.getCounters().islandCount).toBe(1);

        // A static body creates no island.
        const stat = world.createBody({});
        stat.createHull({}, makeBoxHull(1, 1, 1));
        expect(world.getCounters().islandCount).toBe(1);

        dyn.destroy();
        expect(world.getCounters().islandCount).toBe(0);
        world.destroy();
    });

    test("a height field on a static body attaches; on a dynamic body it is rejected", () => {
        const world = new World();
        const hf = () => createGrid(4, 4, { x: 1, y: 1, z: 1 }, false);

        const stat = world.createBody({});
        expect(stat.createHeightField({}, hf()).isValid()).toBe(true);

        // Height fields carry no mass, so the C rejects them on a non-static body (else zero mass → NaN).
        const dyn = world.createBody({ type: BodyType.Dynamic });
        expect(() => dyn.createHeightField({}, hf())).toThrow();

        world.destroy();
    });
});

describe("world gravity accessor", () => {
    // Both setGravity and getGravity copy the vector, so neither aliases the caller's object.
    test("get/set round-trips and neither side aliases the caller", () => {
        const world = new World();
        const g = { x: 1, y: -20, z: 3 };
        world.setGravity(g);
        g.y = 999; // mutating the input after set must not leak into the world
        expect(world.getGravity()).toEqual({ x: 1, y: -20, z: 3 });

        const out = world.getGravity();
        out.x = 777; // mutating the returned copy must not corrupt the world
        expect(world.getGravity().x).toBe(1);
        world.destroy();
    });
});

const DT = 1 / 60;
const IDENTITY_Q = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const len = (v: Vec3): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

describe("forces and impulses", () => {
    // A free unit-density sphere in a gravity-free world, kept awake so the state persists.
    function freeSphere(maximumLinearSpeed?: number) {
        const world = new World({
            gravity: { x: 0, y: 0, z: 0 },
            enableSleep: false,
            ...(maximumLinearSpeed !== undefined ? { maximumLinearSpeed } : {}),
        });
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        body.applyMassFromShapes();
        return { world, body };
    }

    test("linear impulse to center changes velocity by J/m, with no spin", () => {
        const { world, body } = freeSphere();
        const m = body.getMass();
        body.applyLinearImpulseToCenter({ x: 2, y: 0, z: 0 });
        const v = body.getLinearVelocity();
        expect(v.x).toBeCloseTo(2 / m, 5);
        expect(len(v)).toBeCloseTo(2 / m, 5);
        expect(len(body.getAngularVelocity())).toBeLessThan(1e-6);
        world.destroy();
    });

    test("off-center linear impulse also spins the body", () => {
        const { world, body } = freeSphere();
        // +y impulse at +x offset: torque = r × J = (0.5,0,0) × (0,1,0) = (0,0,0.5) → spin about +z.
        body.applyLinearImpulse({ x: 0, y: 1, z: 0 }, { x: 0.5, y: 0, z: 0 });
        const w = body.getAngularVelocity();
        expect(w.z).toBeGreaterThan(1e-4);
        expect(Math.abs(w.x)).toBeLessThan(1e-6);
        expect(Math.abs(w.y)).toBeLessThan(1e-6);
        world.destroy();
    });

    test("angular impulse changes angular velocity by J/I", () => {
        const { world, body } = freeSphere();
        const inertia = body.getMassData().inertia.cx.x; // sphere: isotropic diagonal
        body.applyAngularImpulse({ x: 3, y: 0, z: 0 });
        const w = body.getAngularVelocity();
        expect(w.x).toBeCloseTo(3 / inertia, 4);
        expect(Math.abs(w.y)).toBeLessThan(1e-6);
        expect(Math.abs(w.z)).toBeLessThan(1e-6);
        world.destroy();
    });

    test("linear speed clamps to the world maximum, keeping direction", () => {
        const maxSpeed = 50;
        const { world, body } = freeSphere(maxSpeed);
        body.applyLinearImpulseToCenter({ x: 1e9, y: 0, z: 0 });
        const v = body.getLinearVelocity();
        expect(len(v)).toBeCloseTo(maxSpeed, 3);
        expect(v.x).toBeCloseTo(maxSpeed, 3);
        world.destroy();
    });

    test("force to center integrates to v = (F/m)·dt over one step", () => {
        const { world, body } = freeSphere();
        const m = body.getMass();
        body.applyForceToCenter({ x: 10, y: 0, z: 0 });
        world.step(DT, 1);
        expect(body.getLinearVelocity().x).toBeCloseTo((10 / m) * DT, 4);
        world.destroy();
    });

    test("torque integrates to w = (T/I)·dt over one step", () => {
        const { world, body } = freeSphere();
        const inertia = body.getMassData().inertia.cx.x;
        body.applyTorque({ x: 0, y: 0, z: 2 });
        world.step(DT, 1);
        expect(body.getAngularVelocity().z).toBeCloseTo((2 / inertia) * DT, 4);
        world.destroy();
    });

    test("off-center force also produces torque", () => {
        const { world, body } = freeSphere();
        body.applyForce({ x: 0, y: 5, z: 0 }, { x: 0.5, y: 0, z: 0 });
        world.step(DT, 1);
        expect(body.getAngularVelocity().z).toBeGreaterThan(1e-4);
        world.destroy();
    });

    test("wake=false leaves a sleeping body asleep; wake=true wakes it", () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const body = world.createBody({ type: BodyType.Dynamic, isAwake: false });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        body.applyMassFromShapes();
        expect(body.isAwake()).toBe(false);

        body.applyForceToCenter({ x: 10, y: 0, z: 0 }, false);
        expect(body.isAwake()).toBe(false);

        body.applyForceToCenter({ x: 10, y: 0, z: 0 }, true);
        expect(body.isAwake()).toBe(true);
        world.destroy();
    });
});

describe("setTransform", () => {
    test("teleports position and rotation", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        body.applyMassFromShapes();
        const s = Math.sin(Math.PI / 4);
        body.setTransform({ x: 3, y: 4, z: 5 }, { v: { x: 0, y: 0, z: s }, s });
        expect(body.getPosition()).toEqual({ x: 3, y: 4, z: 5 });
        expect(body.getRotation().s).toBeCloseTo(s, 6);
        expect(body.getRotation().v.z).toBeCloseTo(s, 6);
        world.destroy();
    });

    test("carries the center of mass with the body", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({ density: 1 }, { center: { x: 1, y: 0, z: 0 }, radius: 0.5 });
        body.applyMassFromShapes();
        body.setTransform({ x: 10, y: 0, z: 0 }, IDENTITY_Q);
        expect(body.getWorldCenterOfMass().x).toBeCloseTo(11, 5); // origin 10 + local COM 1
        world.destroy();
    });

    test("relocates the broadphase proxy so the body collides at its new location", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
        // Ground under x = 100, top face at y = 0.
        const ground = world.createBody({ position: { x: 100, y: -1, z: 0 } });
        ground.createHull({}, makeBoxHull(5, 1, 5));

        const ball = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        ball.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        ball.setTransform({ x: 100, y: 2, z: 0 }, IDENTITY_Q);

        for (let i = 0; i < 60; ++i) world.step(DT, 4);

        // Settled on the relocated ground, not tunneled through (which is what a stale proxy causes).
        const p = ball.getPosition();
        expect(Math.abs(p.x - 100)).toBeLessThan(0.5);
        expect(p.y).toBeGreaterThan(-0.1);
        world.destroy();
    });
});

describe("setType", () => {
    test("dynamic → static freezes the body", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        for (let i = 0; i < 10; ++i) world.step(DT, 4);
        expect(body.getLinearVelocity().y).toBeLessThan(0); // falling

        body.setType(BodyType.Static);
        expect(body.getType()).toBe(BodyType.Static);
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });

        const y = body.getPosition().y;
        for (let i = 0; i < 20; ++i) world.step(DT, 4);
        expect(body.getPosition().y).toBe(y); // no longer integrates
        world.destroy();
    });

    test("static → dynamic makes the body fall and gives it an island", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
        const body = world.createBody({ position: { x: 0, y: 5, z: 0 } }); // static default
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        expect(world.getCounters().islandCount).toBe(0);

        body.setType(BodyType.Dynamic);
        expect(body.getType()).toBe(BodyType.Dynamic);
        expect(world.getCounters().islandCount).toBe(1);

        const y0 = body.getPosition().y;
        for (let i = 0; i < 30; ++i) world.step(DT, 4);
        expect(body.getPosition().y).toBeLessThan(y0 - 0.5);
        world.destroy();
    });

    test("setType to the same type is a no-op", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        const islandCount = world.getCounters().islandCount;

        body.setType(BodyType.Dynamic);
        expect(body.getType()).toBe(BodyType.Dynamic);
        expect(world.getCounters().islandCount).toBe(islandCount);
        expect(body.isValid()).toBe(true);
        world.destroy();
    });

    test("setType to non-static is rejected for a compound-shape body", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
        const mat = defaultSurfaceMaterial();
        const compound = createCompound({
            hulls: [
                {
                    hull: makeBoxHull(1, 0.25, 1),
                    transform: { p: { x: 0, y: 0, z: 0 }, q: IDENTITY_Q },
                    material: mat,
                },
            ],
        });
        const body = world.createBody({}); // static
        body.createCompound({}, compound!);

        body.setType(BodyType.Dynamic);
        expect(body.getType()).toBe(BodyType.Static); // rejected, unchanged

        // The world is still usable (the rejection released its lock).
        const other = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        other.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world.step(DT, 4);
        expect(other.isValid()).toBe(true);
        world.destroy();
    });

    test("changing type on a jointed body keeps the joint and world consistent", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
        const a = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        a.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 1.5, y: 5, z: 0 } });
        b.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
        const joint = world.createDistanceJoint(a, b, { length: 1.5 });

        // Pin `a` by making it static; `b` should hang from it through the joint (transferJoint path).
        a.setType(BodyType.Static);
        for (let i = 0; i < 60; ++i) world.step(DT, 4);

        expect(a.getPosition()).toEqual({ x: 0, y: 5, z: 0 }); // static: fixed
        const pb = b.getPosition();
        expect(Number.isFinite(pb.x) && Number.isFinite(pb.y) && Number.isFinite(pb.z)).toBe(true);
        // Rigid distance joint holds |b - a| ≈ 1.5; allow slack for solver softness.
        const d = Math.sqrt(pb.x * pb.x + (pb.y - 5) * (pb.y - 5) + pb.z * pb.z);
        expect(d).toBeLessThan(2.0);
        expect(joint.isValid()).toBe(true);
        world.destroy();
    });
});

describe("setAwake", () => {
    test("setAwake(false) sleeps an awake body; setAwake(true) wakes it", () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        expect(body.isAwake()).toBe(true);

        body.setAwake(false);
        expect(body.isAwake()).toBe(false);

        body.setAwake(true);
        expect(body.isAwake()).toBe(true);
        world.destroy();
    });

    test("setAwake(true) wakes a body created asleep", () => {
        const world = new World({ gravity: { x: 0, y: 0, z: 0 } });
        const body = world.createBody({ type: BodyType.Dynamic, isAwake: false });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        expect(body.isAwake()).toBe(false);

        body.setAwake(true);
        expect(body.isAwake()).toBe(true);
        world.destroy();
    });
});

// Body state is column-resident in a singleton region shared by every world, so two live worlds can't
// be stepped interleaved. The guard evicts the previous owner on each step; a sequential hand-off keeps
// working, but re-stepping a world after another took the region over throws.
describe("single live world guard", () => {
    const spawn = () => {
        const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: false });
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        return world;
    };

    test("re-stepping a world after another took the region over throws", () => {
        const a = spawn();
        const b = spawn();
        a.step(1 / 60, 4); // a claims the region
        b.step(1 / 60, 4); // b takes it over, evicting a
        expect(() => a.step(1 / 60, 4)).toThrow(/two live worlds/);
        a.destroy();
        b.destroy();
    });

    test("sequential worlds (each abandoned before the next steps) keep working", () => {
        const a = spawn();
        a.step(1 / 60, 4);
        a.step(1 / 60, 4); // same owner re-steps fine
        a.destroy();

        const b = spawn();
        expect(() => b.step(1 / 60, 4)).not.toThrow();
        b.destroy();
    });
});

describe("f32 API boundary rounding", () => {
    // Every user float rounds to f32 once at ingress (createWorld/createBody/createShape + runtime
    // setters), so internal arithmetic always starts from f32 values — the bit-exact parity contract.
    test("shape density rounds to f32 at ingress (mass derives from fround(density))", () => {
        const density = 1234.5678; // not f32-representable — an unrounded f64 would reach mass compute
        const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 0.5 }; // 0.5 is exact → roundSphere is a no-op
        const world = new World();
        const body = world.createBody({ type: BodyType.Dynamic });
        body.createSphere({ density }, sphere);
        const rounded = computeSphereMass(sphere, f32(density)).mass;
        const unrounded = computeSphereMass(sphere, density).mass;
        // Mutation guard: the two paths genuinely diverge, so the assertion below has teeth.
        expect(rounded).not.toBe(unrounded);
        expect(body.getMass()).toBe(rounded);
        world.destroy();
    });

    test("world gravity rounds to f32 (def ingress and setGravity)", () => {
        const gy = -9.81; // not f32-representable
        expect(f32(gy)).not.toBe(gy); // mutation guard: the value actually needs rounding
        const world = new World({ gravity: { x: 0, y: gy, z: 0 } });
        expect(world.getGravity().y).toBe(f32(gy));
        world.setGravity({ x: 0, y: gy, z: 0 });
        expect(world.getGravity().y).toBe(f32(gy));
        world.destroy();
    });
});
