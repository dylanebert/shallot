import { expect, test } from "bun:test";
import { Scheduler, Time } from "./scheduler";
import { World } from "./world";

test("scheduler fixed and frame paths require no placement or other world domain", () => {
    const scheduler = new Scheduler();
    const world = new Proxy({} as World, {
        get(_target, key) {
            throw new Error(`scheduler accessed world domain ${String(key)}`);
        },
    });
    const groups: string[] = [];
    for (const group of ["fixed", "simulation", "draw"] as const)
        scheduler.register({
            group,
            update: (w) => {
                expect(w).toBe(world);
                groups.push(group);
            },
        });
    scheduler.tick(world);
    scheduler.step(world, { deltaTime: Time.FIXED_DT });
    expect(groups).toEqual(["fixed", "fixed", "simulation", "draw"]);
});

test("catch-up ticks see integer tick time; zero-tick frames keep virtual time", () => {
    const world = new World();
    const elapsed: number[] = [];
    world.addSystem({
        group: "fixed",
        update: (w) => {
            expect(w.time.deltaTime).toBe(Time.FIXED_DT);
            elapsed.push(w.time.elapsed);
        },
    });
    world.step(Time.FIXED_DT * 3.5);
    expect(elapsed).toEqual([1, 2, 3].map((n) => n * Time.FIXED_DT));
    expect(world.time.elapsed).toBe(Time.FIXED_DT * 3.5);
    world.step(0);
    expect(elapsed).toHaveLength(3);
});

test("exact ticks ignore pause and scale, leave pacing and sibling clocks alone, and run only fixed", () => {
    const a = new World();
    const b = new World();
    const groups: string[] = [];
    for (const group of ["setup", "fixed", "simulation", "draw"] as const)
        a.addSystem({
            group,
            update: () => {
                groups.push(group);
            },
        });
    a.pause();
    a.setTimeScale(0);
    for (let i = 0; i < 8; i++) a.tick();
    expect(groups).toEqual(Array(8).fill("fixed"));
    expect(a.time.fixedTick).toBe(8);
    expect(a.time.elapsed).toBe(0);
    expect(a.time.fixedSteps).toBe(0);
    expect(b.time.fixedTick).toBe(0);
    b.tick();
    expect(b.time.fixedTick).toBe(1);
    a.step();
    expect(a.time.fixedTick).toBe(8);
    a.resume();
    a.setTimeScale(2);
    a.step();
    expect(a.time.fixedTick).toBe(10);
});

test("tick refuses reentry during step and tick and restores clocks after named failures", () => {
    for (const drive of ["step", "tick"] as const) {
        const world = new World();
        const nested = { name: "nested", group: "fixed" as const, update: (w: World) => w.tick() };
        world.addSystem(nested);
        expect(() => world[drive]()).toThrow(/nested.*inside a step or tick/);
        expect(world.time.elapsed).toBe(drive === "step" ? Time.FIXED_DT : 0);
        world.removeSystem(nested);
        world.tick();
        expect(world.time.fixedTick).toBe(2);
    }
});

test("n exact ticks then draw equal frame-driven ticks in fixed values", () => {
    const a = new World();
    const b = new World();
    let x = 0;
    let y = 0;
    a.addSystem({
        group: "fixed",
        update: (w) => {
            x += w.time.elapsed;
        },
    });
    b.addSystem({
        group: "fixed",
        update: (w) => {
            y += w.time.elapsed;
        },
    });
    for (let i = 0; i < 120; i++) {
        a.tick();
        b.step();
    }
    a.step(0);
    expect(x).toBe(y);
    expect(a.time.fixedTick).toBe(b.time.fixedTick);
});
