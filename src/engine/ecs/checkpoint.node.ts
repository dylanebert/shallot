import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";
import { attachCanvas, Camera, RenderingPlugin } from "../../core/rendering";
import { createApp } from "../app";
import { CanvasContext } from "../app/canvas.fixture";
import { GlobalTransform, globalTransformTable, probeBuffer, Time, Transform } from "../index";

setDefaultTimeout(CEILING.node);
await setupGlobals();
if (typeof ResizeObserver === "undefined")
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });

test("built world recovery publishes restored fields and discards old interpolation without altering pacing or GPU frame", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    const { world } = app;
    try {
        let context: CanvasContext;
        const canvas = {
            width: 32,
            height: 24,
            style: { imageRendering: "auto" },
            getContext: () => context,
            getBoundingClientRect: () => ({ width: 32, height: 24 }),
        } as unknown as HTMLCanvasElement;
        context = new CanvasContext(canvas, 32, 24);
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.storage(Transform).translation.set(camera, 0, 0, 5, 0);
        attachCanvas(camera, canvas, world);
        const eid = world.create();
        world.add(eid, Transform);
        const initial = world.create();
        world.add(initial, Transform);
        const identity = world.ref(initial);
        let transient = initial;
        const accessor = world.storage(GlobalTransform).translation.x;
        world.addSystem({
            group: "fixed",
            update: (w) => {
                w.storage(Transform).translation.x.set(eid, w.time.fixedTick * 10);
                w.destroy(transient);
                transient = w.create();
                w.add(transient, Transform, { translation: [w.time.elapsed, 0, 0, 0] });
            },
        });
        const table = globalTransformTable(world);
        world.step(Time.FIXED_DT * 2.5);
        const checkpoint = world.checkpoint();
        const reference = world.ref(transient);
        const retained = world.query([Transform]);
        const membership = [...retained];
        world.tick();
        world.tick();
        const continuation = {
            entities: world.entities(),
            members: [...retained],
            refs: world.entities().map((eid) => world.ref(eid)),
            fields: [...world.storage(Transform).translation.column],
        };
        world.step(0);
        const frame = world.gpu.frame;
        const pacing = { ...world.time };
        world.restore(checkpoint);
        expect(accessor.get(eid)).toBe(20);
        expect(world.resolve(identity)).toBe(0);
        expect(world.resolve(reference)).toBe(transient);
        expect([...retained]).toEqual(membership);
        expect(world.gpu.frame).toBe(frame);
        expect({ ...world.time, fixedTick: pacing.fixedTick }).toEqual(pacing);
        world.step(0);
        const words = new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(20, 5);
        world.tick();
        world.step(0);
        const replay = new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(replay[table.rowIndex(eid) * 12]).toBeCloseTo(25, 5);
        world.tick();
        expect({
            entities: world.entities(),
            members: [...retained],
            refs: world.entities().map((eid) => world.ref(eid)),
            fields: [...world.storage(Transform).translation.column],
        }).toEqual(continuation);
    } finally {
        app.dispose();
    }
});
