import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import { BodyType, defaultShapeDef } from "./common/types";
import { init, kernel, kernelState, shutdown } from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import { createRock, makeBoxHull } from "./shapes/build-hull";
import { createGridMesh, createWave, createWaveMesh } from "./shapes/build-nonconvex";
import { createHeightField } from "./shapes/heightfield";
import { createMesh } from "./shapes/mesh";
import {
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "./shapes/shape";
import { createFilterJoint, defaultJointDef } from "./solver/joint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./solver/revoluteJoint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { bodySetAwake } from "./world/body";
import { hashWorldState } from "./world/hash";
import type { WorldState } from "./world/world";

setDefaultTimeout(TIMEOUT.node);

function scene(owner: World, variant: number): { world: WorldState; sleeper: number } {
    const world = createTestWorld({}, owner);
    const ground = createTestBody(world, { type: BodyType.Static });
    createHullShape(world, ground, defaultShapeDef(), makeBoxHull(12, 0.25 + variant * 0.1, 12));
    const sphere = (x: number, y: number, z = 0) => {
        const body = createTestBody(world, { type: BodyType.Dynamic, position: { x, y, z } });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        return body;
    };
    sphere(0, 0.75 + variant * 0.1);
    sphere(0, 1.75 + variant * 0.1);
    const sleeper = sphere(3, 0.75 + variant * 0.1);
    bodySetAwake(world, sleeper, false);
    const filteredA = sphere(-3, 2);
    const filteredB = sphere(-3, 2.25);
    createFilterJoint(world, {
        ...defaultJointDef(),
        bodyIdA: filteredA,
        bodyIdB: filteredB,
    });
    const arm = sphere(6, 3);
    createRevoluteJoint(world, {
        ...defaultRevoluteJointDef({
            ...defaultJointDef(),
            bodyIdA: ground,
            bodyIdB: arm,
            localFrameA: {
                p: { x: 6, y: 3, z: 0 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            },
        }),
        enableMotor: true,
        maxMotorTorque: 10,
        motorSpeed: 1 + variant,
    });
    const hull = createTestBody(world, {
        type: BodyType.Dynamic,
        position: { x: 1, y: 4, z: 1 },
    });
    createHullShape(world, hull, testShapeDef(), makeBoxHull(0.4, 0.6, 0.3));
    const meshBody = createTestBody(world, {
        type: BodyType.Static,
        position: { x: 0, y: 1, z: 5 },
    });
    const mesh = createMesh({
        vertices: [
            { x: -2, y: 0, z: -2 },
            { x: 2, y: 0, z: -2 },
            { x: 2, y: 0, z: 2 },
            { x: -2, y: 0, z: 2 },
        ],
        indices: [0, 2, 1, 0, 3, 2],
    })!;
    createMeshShape(world, meshBody, defaultShapeDef(), mesh, { x: 1, y: 1, z: 1 });
    sphere(0, 1.5, 5);
    const terrain = createTestBody(world, {
        type: BodyType.Static,
        position: { x: -6, y: 1, z: 5 },
    });
    const heightField = createHeightField({
        heights: [0, 0, 0, 0],
        materialIndices: null,
        scale: { x: 3, y: 1, z: 3 },
        countX: 2,
        countZ: 2,
        globalMinimumHeight: 0,
        globalMaximumHeight: 1,
        clockwiseWinding: false,
    });
    createHeightFieldShape(world, terrain, defaultShapeDef(), heightField);
    sphere(-5, 1.5, 6);
    const bullet = createTestBody(world, {
        type: BodyType.Dynamic,
        position: { x: 0, y: 8, z: 2 },
        isBullet: true,
        linearVelocity: { x: 0, y: -100, z: 0 },
    });
    createSphereShape(world, bullet, testShapeDef(), {
        center: { x: 0, y: 0, z: 0 },
        radius: 0.1,
    });
    return { world, sleeper };
}

function tick(subject: ReturnType<typeof scene>, index: number, ray = true): bigint {
    if (index === 30) bodySetAwake(subject.world, subject.sleeper, true);
    if (ray) {
        const query = queryColumns(subject.world);
        const k = query.prepare({ x: 0, y: 10, z: 0 });
        query.translation({ x: 0, y: -20, z: 0 });
        k.worldQuery(subject.world.worldId, 3, 0);
    }
    stepTestWorld(subject.world, 1 / 60);
    return hashWorldState(subject.world);
}

test("hull authoring uses its owning World's kernel and survives sibling builds and scratch growth", async () => {
    const a = new World();
    const b = new World();
    const standalone = kernelState(undefined).instance;
    try {
        await init(a, { threads: 0 });
        await init(b, { threads: 0 });
        expect(kernel(a)).not.toBe(kernel(b));
        const first = makeBoxHull(1, 2, 3, a);
        const snapshot = structuredClone(first);
        createRock(2, b);
        kernel(a).hullBuildInputPtr(100_000);
        expect(makeBoxHull(1, 2, 3, a)).toEqual(snapshot);
        expect(first).toEqual(snapshot);
        expect(kernelState(undefined).instance).toBe(standalone);
        await shutdown(a);
        expect(makeBoxHull(1, 2, 3, b)).toEqual(snapshot);
    } finally {
        await shutdown(a);
        await shutdown(b);
        a.dispose();
        b.dispose();
    }
});

test("mesh and height-field authoring use their owning kernel and preserve independent outputs across scratch growth", async () => {
    const a = new World(),
        b = new World();
    const standalone = kernelState(undefined).instance;
    try {
        await init(a, { threads: 0 });
        await init(b, { threads: 0 });
        const mesh = createWaveMesh(6, 5, 1, 0.4, 0.05, 0.1, a);
        const field = createWave(32, 32, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false, a);
        const expectedMesh = structuredClone(mesh),
            expectedField = structuredClone(field);
        createGridMesh(4, 5, 0.3, 3, true, b);
        kernel(a).nonconvexBuildInputPtr(1_000_000);
        expect(createWaveMesh(6, 5, 1, 0.4, 0.05, 0.1, a)).toEqual(expectedMesh);
        expect(createWave(32, 32, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false, a)).toEqual(
            expectedField,
        );
        expect(mesh).toEqual(expectedMesh);
        expect(field).toEqual(expectedField);
        expect(kernelState(undefined).instance).toBe(standalone);
        await shutdown(a);
        expect(createWaveMesh(6, 5, 1, 0.4, 0.05, 0.1, b)).toEqual(expectedMesh);
        expect(createWave(32, 32, { x: 2, y: 1.5, z: 2 }, 0.1, 0.03333, false, b)).toEqual(
            expectedField,
        );
    } finally {
        await shutdown(a);
        await shutdown(b);
        a.dispose();
        b.dispose();
    }
});

function solo(variant: number): bigint[] {
    const owner = new World();
    const subject = scene(owner, variant);
    try {
        return Array.from({ length: 120 }, (_, i) => tick(subject, i));
    } finally {
        destroyTestWorld(subject.world);
        owner.dispose();
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
                        destroyTestWorld(b.world);
                        b = undefined;
                    }
                    expect(tick(a, i, ray), `target tick ${i}`).toBe(expectedA[i]);
                    if (b) expect(tick(b, i, ray), `sibling tick ${i}`).toBe(expectedB[i]);
                }
            } finally {
                if (b) destroyTestWorld(b.world);
                destroyTestWorld(a.world);
                owner.dispose();
            }
        });
    }
}
