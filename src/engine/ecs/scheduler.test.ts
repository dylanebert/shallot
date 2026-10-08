import { expect, test } from "bun:test";
import { Scheduler } from "./scheduler";
import { World } from "./world";

for (const boundary of ["before", "after"] as const) {
    for (const implicitFirst of [true, false]) {
        test(`${boundary} boundary uniqueness treats an omitted group as simulation (${implicitFirst ? "implicit" : "explicit"} first)`, () => {
            const world = new World();
            const order: string[] = [];
            const first = {
                ...(implicitFirst ? {} : { group: "simulation" as const }),
                boundary,
                update: () => {
                    order.push("first");
                },
            };
            const second = {
                ...(implicitFirst ? { group: "simulation" as const } : {}),
                boundary,
                update: () => {
                    order.push("second");
                },
            };
            try {
                world.addSystem(first);
                expect(() => world.addSystem(second)).toThrow(
                    `System group simulation already has a ${boundary} boundary`,
                );
                expect(world.hasSystem(second)).toBe(false);
                world.step(0);
                expect(order).toEqual(["first"]);
            } finally {
                world.dispose();
            }
        });
    }
}

test("engine group boundaries bracket first, last, late-added and terminal systems without edges", () => {
    const scheduler = new Scheduler();
    const world = new World();
    const order: string[] = [];
    for (const [name, constraints] of [
        ["first", { first: true }],
        ["normal", {}],
        ["last", { last: true }],
        ["terminal", { terminal: true }],
    ] as const)
        scheduler.register({
            group: "fixed",
            ...constraints,
            update: () => {
                order.push(name);
            },
        });
    scheduler.registerBoundary(
        {
            group: "fixed",
            update: () => {
                order.push("start");
            },
        },
        "before",
    );
    scheduler.registerBoundary(
        {
            group: "fixed",
            update: () => {
                order.push("end");
            },
        },
        "after",
    );
    scheduler.tick(world);
    expect(order).toEqual(["start", "first", "normal", "last", "terminal", "end"]);
    order.length = 0;
    scheduler.register({
        group: "fixed",
        last: true,
        update: () => {
            order.push("late");
        },
    });
    scheduler.tick(world);
    expect(order).toEqual(["start", "first", "normal", "last", "late", "terminal", "end"]);
});

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
