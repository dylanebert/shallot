import { expect, test } from "bun:test";
import { createApp, type Plugin } from "../app";
import { component, f32 } from "./component";
import { type SystemGroup, Time } from "./scheduler";
import type { WorldSnapshot } from "./snapshot";

const Position = component("frame-recovery-position", { x: f32 });

async function subject(replay: boolean) {
    let saved: WorldSnapshot;
    let recover = false;
    const captures: number[] = [];
    const marks: number[][] = [];
    const plugin: Plugin = {
        name: "FrameRecovery",
        components: [Position],
        recovery: "stateless",
        systems: [
            {
                group: "preFixed",
                update(w) {
                    if (!recover) return;
                    recover = false;
                    const head = w.time.fixedTick;
                    const alpha = w.time.fixedAlpha;
                    const steps = w.time.fixedSteps;
                    w.restore(saved);
                    while (w.time.fixedTick < head) w.tick();
                    expect(w.time.fixedAlpha).toBe(alpha);
                    expect(w.time.fixedSteps).toBe(steps);
                    // A replay tick inside a CPU frame must not clear the frame's writes.
                    expect(w.fieldStorage(Position, "x").dirty[0]).toBe(2);
                },
            },
            {
                group: "fixed",
                update(w) {
                    const x = w.storage(Position).x;
                    x.set(1, x.get(1) + 1);
                },
            },
            {
                group: "postFixed",
                update(w) {
                    const image = w.snapshot();
                    if (w.time.fixedTick === 1) saved = image;
                    captures.push(w.time.fixedTick);
                },
            },
            {
                group: "simulation",
                update(w) {
                    marks.push([...w.fieldStorage(Position, "x").dirty]);
                },
            },
        ],
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    const w = app.world;
    const eid = w.create();
    expect(eid).toBe(1);
    w.add(eid, Position);
    w.step(Time.FIXED_DT * 2.5);
    return {
        w,
        captures,
        marks,
        recover: () => {
            recover = replay;
        },
    };
}

test("a plugin restores and replays before paced ticks and captures every tick without changing frame results", async () => {
    const direct = await subject(false);
    const replay = await subject(true);
    try {
        for (const [dt, scale] of [
            [Time.FIXED_DT, 1],
            [Time.FIXED_DT, 0.75],
            [1, 4],
        ] as const) {
            for (const s of [direct, replay]) {
                s.recover();
                s.w.setTimeScale(scale);
                s.w.step(dt);
            }
            expect(replay.w.get(1, Position)).toEqual(direct.w.get(1, Position));
            expect(replay.w.time).toEqual(direct.w.time);
            expect(replay.marks).toEqual(direct.marks);
            expect([...replay.w.fieldStorage(Position, "x").dirty]).toEqual([
                ...direct.w.fieldStorage(Position, "x").dirty,
            ]);
        }
        expect(replay.w.time.fixedTick).toBe(8);
        expect(replay.w.time.fixedSteps).toBe(4);
        expect(replay.w.time.throttled).toBe(true);
        expect(replay.captures).toEqual([1, 2, 2, 3, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8]);
    } finally {
        direct.w.dispose();
        replay.w.dispose();
    }
});

test("recovery and nested stepping remain refused outside their system points and in recovery callbacks", async () => {
    let saved: WorldSnapshot;
    const visited: SystemGroup[] = [];
    const groups = ["setup", "preFixed", "fixed", "postFixed", "simulation", "draw"] as const;
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "Refusals",
                recovery: "stateless",
                systems: groups.map((group) => ({
                    group,
                    update(w) {
                        visited.push(group);
                        expect(() => w.step(0)).toThrow("refuses");
                        if (group !== "preFixed") {
                            expect(() => w.tick()).toThrow("refuses");
                            expect(() => w.restore(saved)).toThrow("refuses");
                        }
                        if (group !== "preFixed" && group !== "postFixed")
                            expect(() => w.snapshot()).toThrow("refuses");
                    },
                })),
            },
        ],
    });
    const w = app.world;
    w.registerRecovery("Callbacks", {
        snapshot() {
            expect(() => w.tick()).toThrow("refuses");
            expect(() => w.step()).toThrow("refuses");
            expect(() => w.snapshot()).toThrow("refuses");
            expect(() => w.restore(saved)).toThrow("refuses");
            return 0;
        },
        restore() {
            expect(() => w.tick()).toThrow("refuses");
            expect(() => w.snapshot()).toThrow("refuses");
            expect(() => w.restore(saved)).toThrow("refuses");
        },
    });
    try {
        saved = w.snapshot();
        w.restore(saved);
        w.step();
        expect(visited).toEqual([...groups]);
    } finally {
        w.dispose();
    }
});
