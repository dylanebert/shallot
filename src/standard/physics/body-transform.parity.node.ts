import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { createApp } from "../../engine";
import { getBodyTransform, StandardPhysicsPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const bits = (v: number) => {
    const f = new Float32Array([v]);
    return new Uint32Array(f.buffer)[0]!.toString(16).padStart(8, "0");
};
test("borrowed solver transforms equal native for static, kinematic, dynamic, locked, sleeping and disabled bodies", async () => {
    const gold = (await Bun.file(new URL("./body-transform.gold.txt", import.meta.url)).text())
        .trim()
        .split("\n");
    for (let c = 0; c < 6; c++) {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        const w = app.world;
        try {
            const e = w.create();
            w.add(e, Transform, { translation: [1.25, 3.5, -2.75] });
            w.add(e, Body, {
                type: c === 0 ? BodyType.Static : c === 1 ? BodyType.Kinematic : BodyType.Dynamic,
                angularVelocity: [1, -1, 2],
                linearVelocity: [0.3, -0.2, 0.1],
                motionLocks: c === 3 ? 56 : 0,
                isAwake: Number(c !== 4),
                isEnabled: Number(c !== 5),
            });
            w.add(e, Shape, { kind: ShapeKind.Sphere, radius: 1 });
            expect(getBodyTransform(w, e)).toBeNull();
            let borrowed: ReturnType<typeof getBodyTransform> = null;
            for (let step = 0; step < 3; step++) {
                w.tick();
                const t = getBodyTransform(w, e)!;
                if (borrowed) expect(t).toBe(borrowed);
                borrowed = t;
                const { p, q } = t;
                expect([p.x, p.y, p.z, q.v.x, q.v.y, q.v.z, q.s].map(bits).join(" ")).toBe(
                    gold[c * 3 + step]!,
                );
                const saved = w.snapshot(),
                    expected = structuredClone(t);
                w.tick();
                w.restore(saved);
                expect(getBodyTransform(w, e)).toEqual(expected);
                w.storage(GlobalTransform).translation.set(e, 100, 200, 300);
                w.storage(GlobalTransform).rotation.set(e, 0, 0, 0, 1);
                expect(getBodyTransform(w, e)).toEqual(expected);
                w.restore(saved);
            }
            w.destroy(e);
            expect(getBodyTransform(w, e)).toBeNull();
        } finally {
            app.dispose();
        }
    }
});

test("each World owns its borrowed transform", async () => {
    const apps = await Promise.all(
        [0, 1].map(() => createApp({ defaults: false, plugins: [StandardPhysicsPlugin] })),
    );
    try {
        const records = apps.map((app, i) => {
            const w = app.world,
                e = w.create();
            w.add(e, Transform, { translation: [i, 0, 0] });
            w.add(e, Body);
            w.tick();
            return getBodyTransform(w, e)!;
        });
        expect(records[0]).not.toBe(records[1]);
        expect(records[0]!.p.x).toBe(0);
        expect(records[1]!.p.x).toBe(1);
    } finally {
        for (const app of apps) app.dispose();
    }
});
