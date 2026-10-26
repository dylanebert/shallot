import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    hashPhysics,
    Shape,
    StandardPhysicsPlugin,
    type World,
} from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, profilerBounds, showProfiler } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BenchmarkSamplePlugin } from "../src/benchmarks";
import { EventSamplePlugin } from "../src/events";
import { shapePresentation } from "../src/presentation";
import { profilerFits, sceneSwitcher } from "../src/sample";
import { categories, createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
const benchmarks = categories.find((category) => category.name === "Benchmark")!.scenes;
const plugins = [StandardPhysicsPlugin, EventSamplePlugin, BenchmarkSamplePlugin];
function constructed(world: World) {
    for (const name of ["physics-sync", "world-definition", "constraints"]) {
        StandardPhysicsPlugin.systems!.find((system) => system.name === name)!.update!(world);
    }
    return { hash: hashPhysics(world), entities: world.entities().length };
}
for (const { id } of benchmarks) {
    test(`${id} switched construction equals a fresh build before its first step`, async () => {
        const app = await createApp({ defaults: false, plugins });
        const fresh = await createApp({
            defaults: false,
            plugins,
            setup: (world) => {
                createScene(world, id);
            },
        });
        try {
            const select = sceneSwitcher(app.world);
            select("junkyard");
            app.world.tick();
            select(id);
            expect(constructed(app.world)).toEqual(constructed(fresh.world));
            const types = app.world.storage(Body).type;
            const bodies = app.world.storage(Shape).body;
            for (const eid of app.world.query([Shape])) {
                if (types.get(bodies.get(eid) || eid) === BodyType.Kinematic)
                    expect(shapePresentation(app.world, eid).role).toBe("motorized");
            }
        } finally {
            app.dispose();
            fresh.dispose();
        }
    });
}

test("Benchmark entry shows the profiler, F3 overrides it, and leaving restores either prior state", async () => {
    const window = new EventTarget();
    const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
    Object.defineProperty(globalThis, "window", { configurable: true, value: window });
    const app = await createApp({ defaults: false, plugins });
    try {
        DiagnosticsOverlayPlugin.initialize!(app.world);
        expect(profilerBounds(app.world)).toBeUndefined();
        let fits = true;
        const select = sceneSwitcher(app.world, undefined, () =>
            profilerFits({ left: 0, top: 0, right: 100, bottom: 100 }, [
                { left: 10, top: 10, right: 10, bottom: 10 },
                fits
                    ? { left: 100, top: 0, right: 200, bottom: 100 }
                    : { left: 50, top: 50, right: 150, bottom: 150 },
            ]),
        );
        for (const visible of [false, true]) {
            fits = false;
            showProfiler(app.world, visible);
            select("hull");
            expect(showProfiler(app.world, visible)).toBe(visible);
            select("boxStack");
            expect(showProfiler(app.world, visible)).toBe(visible);
            fits = true;
            showProfiler(app.world, visible);
            select("hull");
            expect(showProfiler(app.world)).toBe(true);
            fits = false;
            select.reconcileProfiler();
            expect(showProfiler(app.world, visible)).toBe(visible);
            fits = true;
            select("hull");
            expect(showProfiler(app.world)).toBe(true);
            const key = new Event("keydown", { cancelable: true });
            Object.assign(key, { key: "F3" });
            window.dispatchEvent(key);
            expect(key.defaultPrevented).toBe(true);
            expect(showProfiler(app.world, false)).toBe(false);
            window.dispatchEvent(key);
            fits = false;
            select.reconcileProfiler();
            expect(showProfiler(app.world, false)).toBe(true);
            select("heightField");
            expect(showProfiler(app.world, visible)).toBe(visible);
            fits = true;
            select("hull");
            expect(showProfiler(app.world, false)).toBe(true);
            select("boxStack");
            expect(showProfiler(app.world, false)).toBe(visible);
        }
    } finally {
        app.dispose();
        if (previous) Object.defineProperty(globalThis, "window", previous);
        else Reflect.deleteProperty(globalThis, "window");
    }
});
