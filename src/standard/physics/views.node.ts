import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import { BodyType, makeBoxHull, PhysicsWorld } from "./api";
import { kernel } from "./kernel/kernel";

setDefaultTimeout(CEILING.node);

test("public body, shape and contact reads survive a sibling's memory growth and destruction without a step or query", () => {
    const owner = new World();
    const a = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    let b: PhysicsWorld | undefined;
    try {
        const mover = a.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = mover.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        a.createBody({ type: BodyType.Static, position: { x: 5, y: 0, z: 0 } }).createHull(
            {},
            makeBoxHull(2, 0.5, 2),
        );
        a.createBody({ type: BodyType.Dynamic, position: { x: 5, y: 1, z: 0 } }).createSphere(
            { enableContactEvents: true },
            { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
        );
        a.step(1 / 60);
        const contact = a.getContactEvents().beginEvents[0].contact;
        const position = mover.getPosition();
        const velocity = mover.getLinearVelocity();
        const bounds = shape.getAABB();
        const mass = shape.computeMassData();
        const manifolds = contact.getData().manifolds;
        expect(manifolds.length).toBeGreaterThan(0);
        const before = kernel(owner).memory.buffer.byteLength;
        b = new PhysicsWorld({}, owner);
        for (let i = 0; i < 2000; i++)
            b.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 2, y: 10, z: 0 },
            }).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        expect(kernel(owner).memory.buffer.byteLength).toBeGreaterThan(before);
        const read = () => {
            expect(mover.getPosition()).toEqual(position);
            expect(mover.getLinearVelocity()).toEqual(velocity);
            expect(shape.getAABB()).toEqual(bounds);
            expect(shape.computeMassData()).toEqual(mass);
            expect(contact.getData().manifolds).toEqual(manifolds);
        };
        read();
        b.destroy();
        b = undefined;
        read();
        mover.setLinearVelocity({ x: 8, y: 9, z: 10 });
        expect(mover.getLinearVelocity()).toEqual({ x: 8, y: 9, z: 10 });
    } finally {
        b?.destroy();
        a.destroy();
    }
});

test("public body, shape and contact reads survive a memory growth that moves no column", () => {
    const owner = new World();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    try {
        const mover = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = mover.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world
            .createBody({ type: BodyType.Static, position: { x: 5, y: 0, z: 0 } })
            .createHull({}, makeBoxHull(2, 0.5, 2));
        world
            .createBody({ type: BodyType.Dynamic, position: { x: 5, y: 1, z: 0 } })
            .createSphere(
                { enableContactEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        world.step(1 / 60);
        const contact = world.getContactEvents().beginEvents[0].contact;
        const position = mover.getPosition();
        const bounds = shape.getAABB();
        const manifolds = contact.getData().manifolds;
        expect(manifolds.length).toBeGreaterThan(0);
        const memory = kernel(owner).memory;
        const buffer = memory.buffer;
        expect(buffer).not.toBeInstanceOf(SharedArrayBuffer);
        memory.grow(1);
        expect(buffer.byteLength).toBe(0);
        expect(mover.getPosition()).toEqual(position);
        expect(shape.getAABB()).toEqual(bounds);
        expect(contact.getData().manifolds).toEqual(manifolds);
    } finally {
        world.destroy();
    }
});

test("a query callback reads a sibling's stale resident views without changing the traversal World", () => {
    const owner = new World();
    const a = new PhysicsWorld({}, owner);
    const b = new PhysicsWorld({}, owner);
    try {
        for (const x of [2, 4, 6])
            a.createBody({ type: BodyType.Static, position: { x, y: 0, z: 0 } }).createSphere(
                {},
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        const body = b.createBody({ type: BodyType.Dynamic, position: { x: 100, y: 0, z: 0 } });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        const expected = body.getPosition();
        a.step(1 / 60);
        const hits: number[] = [];
        a.castRay({ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, (hit) => {
            expect(body.getPosition()).toEqual(expected);
            hits.push(hit.fraction);
            return 1;
        });
        expect(hits).toHaveLength(3);
        for (const [i, hit] of hits.entries()) expect(hit).toBeCloseTo([0.15, 0.35, 0.55][i], 6);
    } finally {
        b.destroy();
        a.destroy();
    }
});
