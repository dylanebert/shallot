import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createApp } from "../../engine";
import { collideSpheres, StandardPhysicsPlugin } from ".";
import { kernel } from "./kernel/kernel";
import { manifoldRecords, manifoldSubject } from "./standalone-manifold.fixture";

setDefaultTimeout(TIMEOUT.node);
test("all nine public manifolds, every field and warm cache equal native touching, speculative and separated records", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        expect(manifoldRecords(app.world).output).toBe(
            await Bun.file(new URL("./standalone-manifold.gold.txt", import.meta.url)).text(),
        );
        const s = manifoldSubject(app.world);
        for (let op = 10; op <= 18; op++) {
            s.simplex.count = 0;
            s.sat.type = 0;
            const boundary =
                op === 15 ? 2.2 : op === 12 || op === 14 ? 1.5 : op === 18 ? 1 : op >= 16 ? 0.5 : 1;
            for (const delta of [-0.25, 0.01, 4]) {
                s.transform.p.y = boundary + delta;
                for (const p of s.triangle) p.y = -s.transform.p.y;
                const m = s.call(op);
                if (delta === 4 || (op <= 11 && delta > 0)) expect(m.pointCount).toBe(0);
                else {
                    expect(m.pointCount).toBeGreaterThan(0);
                    expect(m.points[0].separation > 0).toBe(delta > 0);
                }
            }
        }
    } finally {
        app.dispose();
    }
});
test("collide calls borrow one result per world and reject invalid capacities and caches", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const s = manifoldSubject(app.world),
            t = manifoldSubject(other.world);
        const m = s.call(10),
            p = m.points[0];
        expect(s.call(11)).toBe(m);
        expect(s.call(11).points[0]).toBe(p);
        expect(t.call(10)).not.toBe(m);
        const bytes = kernel(app.world).memory.buffer.byteLength;
        for (let i = 0; i < 1000; i++) {
            for (let op = 10; op <= 18; op++) {
                s.simplex.count = 0;
                s.sat.type = 0;
                expect(s.call(op)).toBe(m);
            }
        }
        expect(kernel(app.world).memory.buffer.byteLength).toBe(bytes);
        kernel(app.world).hullBuildInputPtr(100_000);
        expect(s.call(10)).toBe(m);
        expect(s.call(10).points[0]).toBe(p);
        for (const capacity of [0, 65, 1.5])
            expect(() =>
                collideSpheres(app.world, s.sphere, s.sphere, s.transform, capacity),
            ).toThrow(RangeError);
        s.simplex.count = 5;
        expect(() => s.call(12)).toThrow(RangeError);
        s.sat.type = 9;
        expect(() => s.call(15)).toThrow(RangeError);
    } finally {
        app.dispose();
        other.dispose();
    }
});
