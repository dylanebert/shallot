import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, PhysicsMeshes, Shape, ShapeKind, ShapeMaterials } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine";
import {
    collideBodyMover,
    collideMover,
    hashPhysics,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from ".";

setDefaultTimeout(TIMEOUT.node);
test("world mover returns native traversal planes at capacity, filters live entities and borrows separate World storage", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const origin = { x: 10, y: 2, z: -3 };
        const mover = {
            center1: { x: 0.8, y: -0.2, z: 0 },
            center2: { x: 0.8, y: 0.2, z: 0 },
            radius: 0.5,
        };
        expect(collideMover(world, origin, mover).count).toBe(0);
        const body = world.create();
        world.add(body, Transform, { translation: [10, 2, -3] });
        world.add(body, Body);
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
        const bits = (value: number) =>
            new Uint32Array(new Float32Array([value]).buffer)[0]!.toString(16).padStart(8, "0");
        const rows: string[] = [];
        for (const capacity of [0, 1, 2, 4]) {
            const hits = collideMover(world, origin, mover, capacity);
            let row = `${capacity} ${hits.count}`;
            for (let i = 0; i < hits.count; i++) {
                const hit = hits.records[i]!;
                expect(hit.body).toBe(body);
                const p = hit.plane;
                row += ` ${shapes.indexOf(hit.shape) + 1} ${[p.plane.normal.x, p.plane.normal.y, p.plane.normal.z, p.plane.offset, p.point.x, p.point.y, p.point.z].map(bits).join(" ")} ${p.triangleIndex} ${p.childIndex} ${p.materialIndex}`;
            }
            rows.push(row);
        }
        const ground = world.create();
        world.add(ground, Body);
        const meshShape = world.create();
        const geometry = world.resource(PhysicsMeshes).add({
            name: "two triangles",
            vertices: new Float32Array([-1, 0, -1, -1, 0, 1, 1, 0, 1, 1, 0, -1]),
            indices: new Uint32Array([0, 1, 2, 2, 3, 0]),
            materialIndices: new Uint8Array([0, 1]),
        });
        const materialSet = world.resource(ShapeMaterials).add({
            name: "two materials",
            materials: [0, 1].map(() => ({
                friction: 0.6,
                restitution: 0,
                rollingResistance: 0,
                tangentVelocity: [0, 0, 0] as [number, number, number],
                userMaterialId: 0n,
                customColor: 0,
            })),
        });
        world.add(meshShape, Shape, { body: ground, kind: ShapeKind.Mesh, geometry, materialSet });
        world.tick();
        const meshHits = collideMover(
            world,
            { x: 0, y: 0.4, z: 0 },
            { center1: { x: 0, y: -0.2, z: 0 }, center2: { x: 0, y: 0.2, z: 0 }, radius: 0.5 },
        );
        let meshRow = `mesh ${meshHits.count}`;
        for (let i = 0; i < meshHits.count; i++) {
            const hit = meshHits.records[i]!;
            expect([hit.shape, hit.body]).toEqual([meshShape, ground]);
            const p = hit.plane;
            meshRow += ` ${[p.plane.normal.x, p.plane.normal.y, p.plane.normal.z, p.plane.offset, p.point.x, p.point.y, p.point.z].map(bits).join(" ")} ${p.triangleIndex} ${p.childIndex} ${p.materialIndex}`;
        }
        rows.push(meshRow);
        const hash = hashPhysics(world);
        expect(rows.join("\n") + "\n").toBe(
            await Bun.file(new URL("./world-mover.gold.txt", import.meta.url)).text(),
        );
        const hits = collideMover(world, origin, mover);
        const saved = structuredClone(hits);
        const first = hits.records[0],
            last = hits.records[2];
        const placement = { p: origin, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
        collideBodyMover(world, body, origin, mover, placement);
        collideMover(other.world, origin, mover);
        expect(hits).toEqual(saved);
        expect(collideMover(world, origin, mover, -1).count).toBe(0);
        expect(collideMover(world, origin, mover, 1.9).count).toBe(1);
        expect(collideMover(world, origin, mover, 1)).toBe(hits);
        expect(hits.count).toBe(1);
        expect(hits.records[0]).toBe(first);
        expect(collideMover(world, origin, mover)).toBe(hits);
        expect(hits.records[2]).toBe(last);
        expect(hits).toEqual(saved);
        expect(
            collideMover(world, origin, mover, 32, { categoryBits: 0n, maskBits: 0n }).count,
        ).toBe(0);
        expect(collideMover(world, { x: 100, y: 0, z: 0 }, mover).count).toBe(0);
        expect(hashPhysics(world)).toBe(hash);
        for (const shape of shapes) world.destroy(shape);
        world.destroy(body);
        world.tick();
        expect(collideMover(world, origin, mover).count).toBe(0);
    } finally {
        app.dispose();
        other.dispose();
    }
});
