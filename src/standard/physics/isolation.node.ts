import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import { BodyType, createHeightField, createMesh, hash, makeBoxHull, PhysicsWorld } from "./api";

setDefaultTimeout(CEILING.node);

function scene(owner: World, variant: number) {
    const world = new PhysicsWorld({}, owner);
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(12, 0.25 + variant * 0.1, 12));
    const sphere = (x: number, y: number, z = 0) => {
        const body = world.createBody({ type: BodyType.Dynamic, position: { x, y, z } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        return body;
    };
    sphere(0, 0.75 + variant * 0.1);
    sphere(0, 1.75 + variant * 0.1);
    const sleeper = sphere(3, 0.75 + variant * 0.1);
    sleeper.setAwake(false);
    const filteredA = sphere(-3, 2);
    const filteredB = sphere(-3, 2.25);
    world.createFilterJoint(filteredA, filteredB);
    const arm = sphere(6, 3);
    world.createRevoluteJoint(ground, arm, {
        localFrameA: { p: { x: 6, y: 3, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        enableMotor: true,
        maxMotorTorque: 10,
        motorSpeed: 1 + variant,
    });
    const hull = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 4, z: 1 } });
    hull.createHull({}, makeBoxHull(0.4, 0.6, 0.3));
    const mesh = world.createBody({ type: BodyType.Static, position: { x: 0, y: 1, z: 5 } });
    mesh.createMesh(
        {},
        createMesh({
            vertices: [
                { x: -2, y: 0, z: -2 },
                { x: 2, y: 0, z: -2 },
                { x: 2, y: 0, z: 2 },
                { x: -2, y: 0, z: 2 },
            ],
            indices: [0, 2, 1, 0, 3, 2],
        })!,
    );
    sphere(0, 1.5, 5);
    const terrain = world.createBody({ type: BodyType.Static, position: { x: -6, y: 1, z: 5 } });
    terrain.createHeightField(
        {},
        createHeightField({
            heights: [0, 0, 0, 0],
            materialIndices: null,
            scale: { x: 3, y: 1, z: 3 },
            countX: 2,
            countZ: 2,
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
            clockwiseWinding: false,
        }),
    );
    sphere(-5, 1.5, 6);
    const bullet = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 8, z: 2 },
        isBullet: true,
        linearVelocity: { x: 0, y: -100, z: 0 },
    });
    bullet.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.1 });
    return { world, sleeper };
}

function tick(subject: ReturnType<typeof scene>, index: number, ray = true) {
    if (index === 30) subject.sleeper.setAwake(true);
    if (ray) subject.world.castRayClosest({ x: 0, y: 10, z: 0 }, { x: 0, y: -20, z: 0 });
    subject.world.step(1 / 60);
    return hash(subject.world);
}

function solo(variant: number) {
    const owner = new World();
    const subject = scene(owner, variant);
    try {
        return Array.from({ length: 120 }, (_, i) => tick(subject, i));
    } finally {
        subject.world.destroy();
    }
}

for (const ray of [false, true]) {
    for (const transient of [false, true]) {
        test(`World tick hashes match solo runs ${ray ? "with" : "without"} per-tick rays and ${transient ? "a sibling created, stepped and destroyed between ticks" : "two Worlds stepped alternately"}`, () => {
            const expectedA = solo(0);
            const expectedB = solo(1);
            const owner = new World();
            const a = scene(owner, 0);
            let b = transient ? undefined : scene(owner, 1);
            try {
                for (let i = 0; i < 120; i++) {
                    if (transient) {
                        b = scene(owner, 1);
                        tick(b, 0, ray);
                        b.world.destroy();
                        b = undefined;
                    }
                    expect(tick(a, i, ray), `target tick ${i}`).toBe(expectedA[i]);
                    if (b) expect(tick(b, i, ray), `sibling tick ${i}`).toBe(expectedB[i]);
                }
            } finally {
                b?.world.destroy();
                a.world.destroy();
            }
        });
    }
}
