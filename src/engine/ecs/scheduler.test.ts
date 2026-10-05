import { expect, test } from "bun:test";
import { Scheduler } from "./scheduler";
import { World } from "./world";

for (const phase of ["update", "setup"] as const) {
    test(`a throwing ${phase} ends the step, retries next step and leaves the world disposable`, () => {
        const world = new World();
        const cause = phase === "update" ? new Error("broken") : "broken";
        let attempts = 0;
        let later = 0;
        let disposed = false;
        const broken = {
            name: "spawn",
            [phase]: () => {
                attempts++;
                throw cause;
            },
            dispose: () => {
                disposed = true;
            },
        };
        world.addSystem(broken, "Game");
        world.addSystem({
            after: [broken],
            update: () => {
                later++;
            },
        });
        try {
            for (let i = 1; i <= 2; i++) {
                let error: unknown;
                try {
                    world.step();
                } catch (caught) {
                    error = caught;
                }
                expect(error).toBeInstanceOf(Error);
                expect((error as Error).message).toBe('System "Game/spawn" threw: broken');
                expect((error as Error).cause).toBe(cause);
                expect(attempts).toBe(i);
                expect(later).toBe(0);
            }
        } finally {
            world.dispose();
        }
        expect(disposed).toBe(true);
    });
}

test("the scheduler consumes each updated duration from its lifetime frame input", () => {
    const scheduler = new Scheduler();
    const input = { deltaTime: 0.01 };
    scheduler.step({} as World, input);
    expect(scheduler.time.rawDeltaTime).toBe(0.01);
    input.deltaTime = 0.02;
    scheduler.step({} as World, input);
    expect(scheduler.time.rawDeltaTime).toBe(0.02);
    expect(scheduler.time.elapsed).toBeCloseTo(0.03);
});

test("step refuses a non-finite or negative delta before advancing its clock", () => {
    const scheduler = new Scheduler();
    const before = { ...scheduler.time };
    for (const delta of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
        expect(() => scheduler.step({} as World, { deltaTime: delta })).toThrow(
            "step deltaTime must be a finite, non-negative number",
        );
        expect(scheduler.time).toEqual(before);
    }
});
