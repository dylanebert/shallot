import { expect, test } from "bun:test";
import { type Plugin, swapPlugins } from "../app";
import { Scheduler, type System } from "./scheduler";
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

for (const boundary of ["before", "after"] as const) {
    test(`${boundary} boundary set treats omitted groups as simulation`, () => {
        const world = new World();
        const calls: string[] = [];
        const a: System = {
            boundary,
            update: () => {
                calls.push("a");
            },
        };
        const b: System = {
            boundary,
            group: "simulation",
            before: [a],
            update: () => {
                calls.push("b");
            },
        };
        try {
            world.addSystem(a);
            world.addSystem(b);
            world.step(0);
            expect(calls).toEqual(["b", "a"]);
        } finally {
            world.dispose();
        }
    });
}

test("hot swap pairs every member of an ordered boundary set without duplicating or reordering it", async () => {
    const world = new World();
    const calls: string[] = [];
    const make = (label: string): Plugin => {
        const a: System = {
            group: "fixed",
            boundary: "after",
            name: "a",
            update: () => {
                calls.push(`${label}/a`);
            },
        };
        const b: System = {
            group: "fixed",
            boundary: "after",
            name: "b",
            after: [a],
            update: () => {
                calls.push(`${label}/b`);
            },
        };
        return { name: "BoundarySet", recovery: "stateless", systems: [b, a] };
    };
    const old = make("old"),
        next = make("next");
    try {
        world.registerRecovery(old.name, "stateless");
        for (const system of old.systems!) world.addSystem(system, old.name);
        world.tick();
        expect(await swapPlugins(world, [old], [next])).toEqual({ ok: true });
        world.tick();
        expect(calls).toEqual(["old/a", "old/b", "next/a", "next/b"]);
        for (const system of old.systems!) expect(world.hasSystem(system)).toBe(true);
        for (const system of next.systems!) expect(world.hasSystem(system)).toBe(false);
    } finally {
        world.dispose();
    }
});

for (const boundary of ["before", "after"] as const) {
    test(`${boundary} boundary set follows edges outside every ordinary system`, () => {
        const scheduler = new Scheduler();
        const calls: string[] = [];
        const a: System = {
            boundary,
            group: "fixed",
            update: () => {
                calls.push("a");
            },
        };
        const b: System = {
            boundary,
            group: "fixed",
            after: [a],
            update: () => {
                calls.push("b");
            },
        };
        scheduler.registerBoundary(b, boundary);
        scheduler.registerBoundary(a, boundary);
        for (const [name, flags] of [
            ["first", { first: true }],
            ["normal", {}],
            ["last", { last: true }],
            ["terminal", { terminal: true }],
        ] as const)
            scheduler.register({
                group: "fixed",
                ...flags,
                update: () => {
                    calls.push(name);
                },
            });
        scheduler.tick({} as World);
        const ordinary = ["first", "normal", "last", "terminal"];
        expect(calls).toEqual(
            boundary === "before" ? ["a", "b", ...ordinary] : [...ordinary, "a", "b"],
        );
        calls.length = 0;
        scheduler.register({
            group: "fixed",
            last: true,
            update: () => {
                calls.push("late");
            },
        });
        ordinary.splice(3, 0, "late");
        scheduler.tick({} as World);
        expect(calls).toEqual(
            boundary === "before" ? ["a", "b", ...ordinary] : [...ordinary, "a", "b"],
        );
    });
    test(`${boundary} boundary set refuses cyclic edges`, () => {
        const scheduler = new Scheduler();
        const a: System = { boundary, group: "fixed" };
        const b: System = { boundary, group: "fixed", after: [a], before: [a] };
        scheduler.registerBoundary(a, boundary);
        scheduler.registerBoundary(b, boundary);
        expect(() => scheduler.tick({} as World)).toThrow("Circular dependency");
    });
    test(`${boundary} boundary refuses an edge into the wrong side of ordinary work`, () => {
        const scheduler = new Scheduler();
        const ordinary: System = { group: "fixed" };
        const edge = boundary === "before" ? { after: [ordinary] } : { before: [ordinary] };
        scheduler.register(ordinary);
        scheduler.registerBoundary({ group: "fixed", boundary, ...edge }, boundary);
        expect(() => scheduler.tick({} as World)).toThrow("Unsatisfiable ordering");
    });
}
