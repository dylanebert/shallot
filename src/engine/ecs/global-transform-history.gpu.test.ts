import { expect, setDefaultTimeout, test } from "bun:test";
import { RenderPlugin } from "../../core/rendering";
import {
    build,
    GlobalTransform,
    globalTransformTable,
    probeBuffer,
    Time,
    Transform,
} from "../index";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
function bounded<T>(promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error("GlobalTransform observation exceeded 750 ms")),
            750,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}

test("catch-up ticks retain the penultimate GlobalTransform and no-tick draws advance only interpolation", async () => {
    let eid = -1;
    const app = await build({
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "FixedPlacement",
                systems: [
                    {
                        group: "fixed",
                        update(state) {
                            const input = state.of(Transform);
                            const x = state.time.fixedTick * 10;
                            input.pos.set(eid, x, 2, 3, 0);
                            input.scale.set(eid, x + 1, 1, 1, 0);
                            input.rot.set(eid, 0, 0, 0, state.time.fixedTick % 2 ? -1 : 1);
                        },
                    },
                ],
            },
        ],
        setup(state) {
            eid = state.create();
            state.add(eid, Transform);
        },
    });
    const state = app.state;
    const table = globalTransformTable(state);
    const row = table.rowIndex(eid);
    try {
        state.step(0);
        state.gpu.device.pushErrorScope("validation");
        state.step(Time.FIXED_DT * 2.5);
        let words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[row * 12]).toBeCloseTo(15, 5);
        expect(words[row * 12 + 8]).toBeCloseTo(16, 5);
        expect(words[row * 12 + 7]).toBeCloseTo(1, 5);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(20);
        expect(state.of(GlobalTransform).scale.x.get(eid)).toBe(21);
        state.step(Time.FIXED_DT * 0.25);
        expect(state.time.fixedSteps).toBe(0);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(20);
        words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[row * 12]).toBeCloseTo(17.5, 5);
        expect(words[row * 12 + 8]).toBeCloseTo(18.5, 5);
        expect(() => state.globalTransformRuntime!.previous!.bytes).toThrow("GPU-only");
        expect(await bounded(state.gpu.device.popErrorScope())).toBeNull();
    } finally {
        app.dispose();
    }
});
