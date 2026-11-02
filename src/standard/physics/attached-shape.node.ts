import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expectWarning } from "../../../scripts/test-console";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform, TransformRuntime } from "../../core/transform";
import { createApp } from "../../engine/app";
import { StandardPhysicsPlugin } from ".";
import { setTransform } from "./runtime";

setDefaultTimeout(TIMEOUT.node);

for (const change of ["sensor", "geometry"] as const) {
    test(`same-body ${change} recreation preserves attached placement history`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const w = app.world;
            const body = w.create();
            w.add(body, Transform, { translation: [0, 10, 0] });
            w.add(body, Body, { type: BodyType.Dynamic });
            const shape = w.create();
            w.add(shape, Shape, { body });
            w.tick();
            const previousY = w.storage(GlobalTransform).translation.y.get(body);
            w.resource(TransformRuntime).discontinuities.fill(0);
            if (change === "sensor") w.storage(Shape).isSensor.set(shape, 1);
            else w.storage(Shape).kind.set(shape, ShapeKind.Sphere);
            w.tick();
            expect(w.storage(GlobalTransform).translation.y.get(body)).toBeLessThan(previousY);
            expect(w.get(shape, GlobalTransform)).toEqual(w.get(body, GlobalTransform));
            const discontinuities = w.resource(TransformRuntime).discontinuities;
            expect(discontinuities[body >>> 5] & (1 << (body & 31))).toBe(0);
            expect(discontinuities[shape >>> 5] & (1 << (shape & 31))).toBe(0);
        } finally {
            await app.dispose();
        }
    });
}

test("an existing shape snaps on attachment, rebinding and return to authored placement", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world;
        const bodies = [10, 20].map((x) => {
            const eid = w.create();
            w.add(eid, Transform, { translation: [x, 0, 0] });
            w.add(eid, Body);
            return eid;
        });
        const shape = w.create();
        w.add(shape, Transform, { translation: [99, 0, 0] });
        w.tick(); // Establish the previous placement before any attachment.
        const discontinuities = () => w.resource(TransformRuntime).discontinuities;
        const snaps = (x: number) => {
            expect(w.storage(GlobalTransform).translation.x.get(shape)).toBe(x);
            expect(discontinuities()[shape >>> 5] & (1 << (shape & 31))).not.toBe(0);
        };
        discontinuities().fill(0);
        w.add(shape, Shape, { body: bodies[0] });
        w.tick();
        snaps(10);
        discontinuities().fill(0);
        w.storage(Shape).body.set(shape, bodies[1]);
        w.tick();
        snaps(20);
        discontinuities().fill(0);
        w.remove(shape, Shape);
        w.tick();
        snaps(99);
    } finally {
        await app.dispose();
    }
});

test("attached shapes publish their body's placement and discontinuities, and detach to authored placement", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const w = app.world;
        const body = w.create();
        w.add(body, Transform, { translation: [3, 10, 4], scale: [2, 3, 4] });
        w.add(body, Body, { type: BodyType.Dynamic });
        const shape = w.create();
        w.add(shape, Shape, { body });
        const equal = () => {
            expect(w.has(shape, GlobalTransform)).toBe(true);
            expect(w.get(shape, GlobalTransform)).toEqual(w.get(body, GlobalTransform));
        };
        w.tick();
        equal();
        w.add(shape, Transform, { translation: [99, 98, 97], scale: [5, 6, 7] });
        for (let i = 0; i < 3; i++) {
            w.tick();
            equal();
        }
        w.resource(TransformRuntime).discontinuities.fill(0);
        setTransform(w, body, [7, 20, 9], [0, 0, 0, 1]);
        w.tick();
        equal();
        expect(
            w.resource(TransformRuntime).discontinuities[shape >>> 5] & (1 << (shape & 31)),
        ).not.toBe(0);
        w.remove(shape, Shape);
        w.tick();
        expect(w.get(shape, GlobalTransform)?.translation).toEqual([99, 98, 97]);
        expect(w.get(shape, GlobalTransform)?.scale).toEqual([5, 6, 7]);
        w.add(shape, Shape, { body });
        w.tick();
        equal();
        w.destroy(body);
        expectWarning(/body reference is no longer valid/);
        w.tick();
        expect(w.get(shape, GlobalTransform)?.translation).toEqual([99, 98, 97]);
        expect(w.get(shape, GlobalTransform)?.scale).toEqual([5, 6, 7]);
    } finally {
        await app.dispose();
    }
});
