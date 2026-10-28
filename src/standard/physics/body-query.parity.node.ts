import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import {
    castBodyRay,
    castBodyShape,
    collideBodyMover,
    emptyQueryHit,
    hashPhysics,
    overlapBodyShape,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from ".";

setDefaultTimeout(TIMEOUT.node);
test("body queries use explicit placement and live entities without stepping or changing the world", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    try {
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const body = world.create();
        world.add(body, Transform, { translation: [100, 0, 0] });
        world.add(body, Body, { type: BodyType.Kinematic });
        const shape = world.create();
        world.add(shape, Shape, { body, kind: ShapeKind.Sphere, radius: 1, userMaterialIdLow: 7 });
        const origin = { x: -4, y: 0, z: 0 },
            translation = { x: 8, y: 0, z: 0 };
        const placement = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
        const out = emptyQueryHit();
        const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.25 };
        expect(castBodyRay(world, body, origin, translation, placement, out)).toBe(false);
        world.tick();
        const hash = hashPhysics(world);
        expect(castBodyRay(world, body, origin, translation, placement, out)).toBe(true);
        expect(out).toMatchObject({
            body,
            shape,
            fraction: 0.375,
            userMaterialId: 7n,
            point: { x: -1, y: 0, z: 0 },
            normal: { x: -1, y: 0, z: 0 },
        });
        expect(castBodyRay(world, body, origin, translation, placement, out, undefined, 0.2)).toBe(
            false,
        );
        expect(out).toEqual(emptyQueryHit());
        expect(castBodyShape(world, body, origin, proxy, translation, placement, out)).toBe(true);
        expect([out.body, out.shape]).toEqual([body, shape]);
        expect(overlapBodyShape(world, body, placement.p, proxy, placement)).toBe(true);
        const mover = {
            center1: { x: 0.9, y: -0.2, z: 0 },
            center2: { x: 0.9, y: 0.2, z: 0 },
            radius: 0.25,
        };
        expect(collideBodyMover(world, body, placement.p, mover, placement).count).toBe(1);
        expect(collideBodyMover(world, body, placement.p, mover, placement, 0).count).toBe(0);
        const filter = { categoryBits: 0n, maskBits: 0n };
        expect(castBodyRay(world, body, origin, translation, placement, out, filter)).toBe(false);
        expect(castBodyShape(world, body, origin, proxy, translation, placement, out, filter)).toBe(
            false,
        );
        expect(overlapBodyShape(world, body, placement.p, proxy, placement, filter)).toBe(false);
        expect(collideBodyMover(world, body, placement.p, mover, placement, 4, filter).count).toBe(
            0,
        );
        expect(hashPhysics(world)).toBe(hash);
        world.destroy(body);
        expect(castBodyRay(world, body, origin, translation, placement, out)).toBe(false);
        expect(castBodyShape(world, body, origin, proxy, translation, placement, out)).toBe(false);
        expect(overlapBodyShape(world, body, placement.p, proxy, placement)).toBe(false);
        expect(collideBodyMover(world, body, placement.p, mover, placement).count).toBe(0);
    } finally {
        app.dispose();
    }
});

test("body mover truncates three native planes at capacity and reuses World-owned records", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    try {
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Kinematic });
        const shapes: number[] = [];
        for (let i = 0; i < 3; i++) {
            const shape = world.create();
            world.add(shape, Shape, {
                body,
                kind: ShapeKind.Sphere,
                sphereCenter: [-0.5 + 0.5 * i, 0, 0],
                radius: 1,
            });
            shapes.push(shape);
        }
        world.tick();
        const origin = { x: 10, y: 2, z: -3 };
        const placement = { p: origin, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
        const mover = {
            center1: { x: 0.8, y: -0.2, z: 0 },
            center2: { x: 0.8, y: 0.2, z: 0 },
            radius: 0.5,
        };
        const bits = (value: number) =>
            new Uint32Array(new Float32Array([value]).buffer)[0]!.toString(16).padStart(8, "0");
        const rows: string[] = [];
        for (const capacity of [0, 1, 2, 4]) {
            const hits = collideBodyMover(world, body, origin, mover, placement, capacity);
            let row = `${capacity} ${hits.count}`;
            for (let i = 0; i < hits.count; i++) {
                const hit = hits.records[i]!;
                expect(hit.body).toBe(body);
                const p = hit.plane;
                row += ` ${shapes.indexOf(hit.shape) + 1} ${[p.plane.normal.x, p.plane.normal.y, p.plane.normal.z, p.plane.offset, p.point.x, p.point.y, p.point.z].map(bits).join(" ")} ${p.triangleIndex} ${p.childIndex} ${p.materialIndex}`;
            }
            rows.push(row);
        }
        expect(rows.join("\n") + "\n").toBe(
            await Bun.file(new URL("./body-query-capacity.gold.txt", import.meta.url)).text(),
        );
        const full = collideBodyMover(world, body, origin, mover, placement, 4);
        expect(full.count).toBe(3);
        const first = full.records[0]!;
        const third = full.records[2]!;
        const saved = structuredClone(full);
        expect(collideBodyMover(world, body, origin, mover, placement, 1)).toBe(full);
        expect(full.count).toBe(1);
        expect(full.records[0]).toBe(first);
        expect(collideBodyMover(world, body, origin, mover, placement, 4)).toBe(full);
        expect(full.records[2]).toBe(third);
        expect(full).toEqual(saved);
        world.destroy(body);
        expect(collideBodyMover(world, body, origin, mover, placement)).toBe(full);
        expect(full.count).toBe(0);
    } finally {
        app.dispose();
    }
});
