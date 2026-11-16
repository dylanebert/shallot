import { expect, test } from "bun:test";
import { createApp } from "../app";
import { component, f32 } from "./component";
import { type SystemGroup, Time } from "./scheduler";

const Value = component("JumpValue", { x: f32 });

test("jumpToTick moves only the fixed clock and the following tick uses its new number", async () => {
    let fixed = 0;
    let recovered = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "JumpWitness",
                recovery: () => ({
                    snapshot: () => 0,
                    restore: () => {
                        recovered++;
                    },
                }),
                systems: [
                    {
                        group: "fixed",
                        update: () => {
                            fixed++;
                        },
                    },
                ],
            },
        ],
    });
    const w = app.world;
    try {
        w.step(Time.FIXED_DT * 1.5);
        const eid = w.create();
        w.add(eid, Value, { x: 7 });
        const marks = [...w.fieldStorage(Value, "x").dirty];
        const before = { ...w.time };
        w.jumpToTick(401);
        expect([...w.fieldStorage(Value, "x").dirty]).toEqual(marks);
        expect(w.time.fixedTick).toBe(401);
        expect(fixed).toBe(1);
        expect(recovered).toBe(0);
        expect(w.time.fixedAlpha).toBe(before.fixedAlpha);
        expect(w.time.elapsed).toBe(before.elapsed);
        w.step(Time.FIXED_DT * 0.5);
        expect(w.time.fixedTick).toBe(402);
        expect(fixed).toBe(2);
        for (const tick of [-1, 0.5, NaN, Infinity]) expect(() => w.jumpToTick(tick)).toThrow();
        w.jumpToTick(0);
        w.tick();
        expect(w.time.fixedTick).toBe(1);
    } finally {
        app.dispose();
    }
});

for (const group of [
    "setup",
    "preFixed",
    "fixed",
    "postFixed",
    "simulation",
    "draw",
] as SystemGroup[]) {
    test(`jumpToTick ${group === "preFixed" ? "is allowed" : "refuses"} in ${group}`, async () => {
        let called = false;
        const app = await createApp({
            defaults: false,
            plugins: [
                {
                    name: "JumpTiming",
                    recovery: "stateless",
                    systems: [
                        {
                            group,
                            update(w) {
                                called = true;
                                if (group === "preFixed") w.jumpToTick(50);
                                else expect(() => w.jumpToTick(50)).toThrow();
                            },
                        },
                    ],
                },
            ],
        });
        try {
            app.world.step(Time.FIXED_DT);
            expect(called).toBe(true);
        } finally {
            app.dispose();
        }
    });
}

test("jumpToTick refuses during recovery", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "JumpRecovery",
                recovery: (w) => ({
                    snapshot() {
                        expect(() => w.jumpToTick(5)).toThrow();
                        return 0;
                    },
                    restore() {
                        expect(() => w.jumpToTick(5)).toThrow();
                    },
                }),
            },
        ],
    });
    try {
        app.world.restore(app.world.snapshot());
    } finally {
        app.dispose();
    }
});
