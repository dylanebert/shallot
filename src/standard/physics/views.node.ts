import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, Shape, ShapeKind } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp, type Plugin, type System, World } from "../../engine";
import {
    castShape,
    type PhysicsQueryHit,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from ".";
import { contactIds } from "./collision/contact";
import { readContactManifolds } from "./collision/manifoldstore";
import { BodyType, defaultFilter, defaultSurfaceMaterial } from "./common/types";
import { kernel, rethrowQueryError, setQueryCallback } from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import { readFatAabb, readShapeAabb } from "./kernel/shapecolumns";
import { physicsBodyIndex } from "./runtime";
import { makeBoxHull } from "./shapes/build-hull";
import { createGridMesh } from "./shapes/build-nonconvex";
import { readShapeHull } from "./shapes/hull";
import {
    computeShapeMass,
    createCapsuleShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    getShapeFilter,
    getShapeGeometryScale,
    getShapeMaterial,
    readShapeCapsule,
    readShapeSphere,
    shapeIsValid,
} from "./shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { bodyGetLinearVelocity, bodyGetTransform } from "./world/body";
import { worldCastRay, worldCastShape, worldCollideMover } from "./world/query";
import type { WorldState } from "./world/world";

setDefaultTimeout(TIMEOUT.node);

const identity = () => ({ p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } });

function contact(world: WorldState): number {
    const found = contactIds(world)[0];
    if (found === undefined) throw new Error("view test did not create a contact");
    return found;
}

function reads(world: WorldState, body: number, shape: number, contactId: number): unknown {
    return {
        position: { ...bodyGetTransform(world, body, identity()).p },
        velocity: { ...bodyGetLinearVelocity(world, body, { x: 0, y: 0, z: 0 }) },
        bounds: readShapeAabb(world, shape, {
            lowerBound: { x: 0, y: 0, z: 0 },
            upperBound: { x: 0, y: 0, z: 0 },
        }),
        mass: computeShapeMass(world, shape),
        manifolds: readContactManifolds(world, contactId),
    };
}

test("body, shape and contact reads survive a sibling's memory growth and destruction without a step or query", () => {
    const owner = new World();
    const a = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    let b: WorldState | undefined;
    try {
        const mover = createTestBody(a, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = createSphereShape(a, mover, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        })!;
        const ground = createTestBody(a, { type: BodyType.Static, position: { x: 5, y: 0, z: 0 } });
        createHullShape(a, ground, testShapeDef(), makeBoxHull(2, 0.5, 2));
        const touching = createTestBody(a, {
            type: BodyType.Dynamic,
            position: { x: 5, y: 1, z: 0 },
        });
        createSphereShape(a, touching, testShapeDef({ enableContactEvents: true }), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(a, 1 / 60);
        const contactId = contact(a);
        const before = reads(a, mover, shape, contactId);
        const bytes = kernel(owner).memory.buffer.byteLength;
        b = createTestWorld({}, owner);
        for (let i = 0; i < 2000; i++) {
            const body = createTestBody(b, {
                type: BodyType.Dynamic,
                position: { x: i * 2, y: 10, z: 0 },
            });
            createSphereShape(b, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
        }
        expect(kernel(owner).memory.buffer.byteLength).toBeGreaterThan(bytes);
        expect(reads(a, mover, shape, contactId)).toEqual(before);
        destroyTestWorld(b);
        b = undefined;
        expect(reads(a, mover, shape, contactId)).toEqual(before);
    } finally {
        if (b) destroyTestWorld(b);
        destroyTestWorld(a);
        owner.dispose();
    }
});

test("body, shape and contact reads survive memory growth that moves no column", () => {
    const owner = new World();
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    try {
        const mover = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = createSphereShape(world, mover, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        })!;
        const ground = createTestBody(world, {
            type: BodyType.Static,
            position: { x: 5, y: 0, z: 0 },
        });
        createHullShape(world, ground, testShapeDef(), makeBoxHull(2, 0.5, 2));
        const touching = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 5, y: 1, z: 0 },
        });
        createSphereShape(world, touching, testShapeDef({ enableContactEvents: true }), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(world, 1 / 60);
        const contactId = contact(world);
        const before = reads(world, mover, shape, contactId);
        const memory = kernel(owner).memory;
        const buffer = memory.buffer;
        expect(buffer).not.toBeInstanceOf(SharedArrayBuffer);
        memory.grow(1);
        expect(buffer.byteLength).toBe(0);
        expect(reads(world, mover, shape, contactId)).toEqual(before);
    } finally {
        destroyTestWorld(world);
        owner.dispose();
    }
});

test("a query callback reads a sibling's resident views without changing the traversal World", () => {
    const owner = new World();
    const a = createTestWorld({}, owner);
    const b = createTestWorld({}, owner);
    try {
        for (const x of [2, 4, 6]) {
            const body = createTestBody(a, { type: BodyType.Static, position: { x, y: 0, z: 0 } });
            createSphereShape(a, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
        }
        const body = createTestBody(b, {
            type: BodyType.Dynamic,
            position: { x: 100, y: 0, z: 0 },
        });
        createSphereShape(b, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const expected = { ...bodyGetTransform(b, body, identity()).p };
        stepTestWorld(a, 1 / 60);
        const query = queryColumns(a);
        const k = query.prepare({ x: 0, y: 0, z: 0 });
        query.translation({ x: 10, y: 0, z: 0 });
        const hits: number[] = [];
        const previous = setQueryCallback(owner, (_kind, _id, data) => {
            expect(bodyGetTransform(b, body, identity()).p).toEqual(expected);
            hits.push(data);
            return 1;
        });
        try {
            k.worldQuery(a.worldId, 2, 1);
            rethrowQueryError(owner);
        } finally {
            setQueryCallback(owner, previous);
        }
        expect(hits).toHaveLength(3);
    } finally {
        destroyTestWorld(b);
        destroyTestWorld(a);
        owner.dispose();
    }
});

// Detaches an unshared kernel's buffer without moving any column, so only a staleness check refreshes views.
function detach(owner: World): void {
    const memory = kernel(owner).memory;
    const buffer = memory.buffer;
    expect(buffer).not.toBeInstanceOf(SharedArrayBuffer);
    memory.grow(1);
    expect(buffer.byteLength).toBe(0);
}

test("shape geometry, filter, material, validity and hull reads survive memory growth", () => {
    const owner = new World();
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 } }, owner);
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
        });
        const sphere = createSphereShape(
            world,
            body,
            testShapeDef({
                baseMaterial: { ...defaultSurfaceMaterial(), friction: 0.25, userMaterialId: 7n },
                filter: { ...defaultFilter(), categoryBits: 6n, maskBits: 5n, groupIndex: -3 },
            }),
            { center: { x: 0.5, y: 0, z: 0 }, radius: 0.75 },
        )!;
        const capsule = createCapsuleShape(world, body, testShapeDef(), {
            center1: { x: 0, y: -1, z: 0 },
            center2: { x: 0, y: 1, z: 0 },
            radius: 0.25,
        })!;
        const ground = createTestBody(world, { type: BodyType.Static });
        const hull = createHullShape(world, ground, testShapeDef(), makeBoxHull(2, 0.5, 2))!;
        const mesh = createMeshShape(
            world,
            ground,
            testShapeDef(),
            createGridMesh(2, 2, 1, 0, true),
            {
                x: 2,
                y: 1,
                z: 3,
            },
        )!;
        stepTestWorld(world, 1 / 60);
        const k = kernel(owner);
        const id = (shape: number) => ({
            index1: shape + 1,
            world0: world.worldId,
            generation: k.shapeGeneration(world.worldId, shape),
        });
        const reads = () => {
            return {
                valid: [sphere, capsule, hull, mesh].map((shape) => shapeIsValid(world, id(shape))),
                sphere: readShapeSphere(world, sphere),
                capsule: readShapeCapsule(world, capsule),
                scale: getShapeGeometryScale(world, mesh),
                filter: getShapeFilter(world, sphere),
                material: structuredClone(
                    getShapeMaterial(world, sphere, defaultSurfaceMaterial()),
                ),
                userMaterialId: world.shapeStore.userMaterialIdAt(sphere, 0),
                fat: readFatAabb(world, sphere, {
                    lowerBound: { x: 0, y: 0, z: 0 },
                    upperBound: { x: 0, y: 0, z: 0 },
                }),
                hull: readShapeHull(world, hull).points,
            };
        };
        const before = reads();
        expect(before.material.userMaterialId).toBe(7n);
        detach(owner);
        expect(reads()).toEqual(before);
    } finally {
        destroyTestWorld(world);
        owner.dispose();
    }
});

test("cast and mover callbacks read every hit after a callback grows memory and reads a sibling", () => {
    const owner = new World();
    const a = createTestWorld({}, owner);
    const b = createTestWorld({}, owner);
    try {
        for (const x of [2, 4, 6]) {
            const body = createTestBody(a, { type: BodyType.Static, position: { x, y: 0, z: 0 } });
            createSphereShape(
                a,
                body,
                testShapeDef({
                    baseMaterial: { ...defaultSurfaceMaterial(), userMaterialId: BigInt(x) },
                }),
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        }
        const sibling = createSphereShape(b, createTestBody(b, {}), testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        })!;
        const siblingSphere = readShapeSphere(b, sibling);
        stepTestWorld(a, 1 / 60);
        const origin = { x: 0, y: 0, z: 0 };
        const translation = { x: 10, y: 0, z: 0 };
        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.1 };
        const run = (grow: boolean) => {
            const out: unknown[] = [];
            const siblings: unknown[] = [];
            const visit = () => {
                if (grow && out.length === 1) detach(owner);
                siblings.push(readShapeSphere(b, sibling));
            };
            worldCastRay(a, origin, translation, (hit) => {
                out.push([
                    "ray",
                    hit.shape,
                    hit.fraction,
                    hit.point,
                    hit.normal,
                    hit.userMaterialId,
                ]);
                visit();
                return 1;
            });
            worldCastShape(a, origin, proxy, translation, (hit) => {
                out.push([
                    "shape",
                    hit.shape,
                    hit.fraction,
                    hit.point,
                    hit.normal,
                    hit.userMaterialId,
                ]);
                visit();
                return 1;
            });
            worldCollideMover(
                a,
                { x: 4, y: 0, z: 0 },
                { center1: { x: -2, y: 0, z: 0 }, center2: { x: 2, y: 0, z: 0 }, radius: 0.6 },
                (shape, planes) => {
                    out.push(["mover", shape, planes]);
                    visit();
                    return true;
                },
            );
            return { out, siblings };
        };
        const expected = run(false);
        expect(expected.out).toHaveLength(9);
        expect(expected.siblings).toEqual(Array(9).fill(siblingSphere));
        expect(
            expected.out
                .filter((hit) => (hit as unknown[])[0] === "ray")
                .map((hit) => (hit as unknown[])[5]),
        ).toEqual(expect.arrayContaining([2n, 4n, 6n]));
        expect(run(true)).toEqual(expected);
    } finally {
        destroyTestWorld(b);
        destroyTestWorld(a);
        owner.dispose();
    }
});

test("public cast hits read every hit after a callback grows memory", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).workerCount = 0;
        },
    });
    try {
        const world = app.world;
        for (const x of [2, 4, 6]) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [x, 0, 0] });
            world.add(eid, Body);
            world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
        }
        world.tick();
        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.1 };
        const run = (grow: boolean) => {
            const out: unknown[] = [];
            castShape(
                world,
                { x: 0, y: 0, z: 0 },
                proxy,
                { x: 10, y: 0, z: 0 },
                (hit: PhysicsQueryHit) => {
                    out.push(structuredClone(hit));
                    if (grow && out.length === 1) detach(world);
                    return 1;
                },
            );
            return out;
        };
        const expected = run(false);
        expect(expected).toHaveLength(3);
        expect(run(true)).toEqual(expected);
    } finally {
        app.dispose();
    }
});

test("a teleport that grows memory leaves the pose re-check reading later bodies' current poses", async () => {
    const eids: number[] = [];
    let write = false;
    const writer: System = {
        name: "teleport-before-step",
        group: "fixed",
        before: [StepPhysicsSystem],
        update(world) {
            if (write) world.storage(Transform).translation.set(eids[0], 20, 0, 0);
        },
    };
    const plugin: Plugin = {
        name: "Teleport",
        dependencies: [StandardPhysicsPlugin],
        recovery: "stateless",
        systems: [writer],
    };
    const app = await createApp({
        defaults: false,
        plugins: [plugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).workerCount = 0;
        },
    });
    try {
        const world = app.world;
        for (const x of [2, 4, 6]) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [x, 0, 0] });
            world.add(eid, Body);
            world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
            eids.push(eid);
        }
        // Each published body is re-checked next tick, the teleported one first.
        world.tick();
        const k = kernel(world);
        const setPose = k.bodySetPose;
        const setter = spyOn(k, "bodySetPose").mockImplementation((...args) => {
            setPose(...args);
            detach(world);
        });
        try {
            write = true;
            world.tick();
            expect(setter).toHaveBeenCalledTimes(1);
            expect(setter.mock.calls[0][1]).toBe(physicsBodyIndex(world, eids[0]));
        } finally {
            setter.mockRestore();
        }
    } finally {
        app.dispose();
    }
});
