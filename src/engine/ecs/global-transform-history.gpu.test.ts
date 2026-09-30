import { expect, setDefaultTimeout, test } from "bun:test";
import { attachCanvas, Camera, PointLight, RenderPlugin } from "../../core/rendering";
import { CanvasContext } from "../app/canvas.fixture";
import {
    build,
    GlobalTransform,
    globalTransformTable,
    probeBuffer,
    type State,
    Time,
    Transform,
} from "../index";

setDefaultTimeout(1000);
if (typeof ResizeObserver === "undefined") {
    Object.assign(globalThis, {
        ResizeObserver: class {
            observe() {}
            unobserve() {}
            disconnect() {}
        },
    });
}
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
function attachTestCamera(state: State): void {
    let context: CanvasContext;
    const canvas = {
        width: 32,
        height: 24,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 32, height: 24 }),
    } as unknown as HTMLCanvasElement;
    context = new CanvasContext(canvas, 32, 24);
    const camera = state.create();
    state.add(camera, Transform);
    state.add(camera, Camera);
    state.of(Transform).pos.set(camera, 0, 0, 5, 0);
    attachCanvas(camera, canvas, state);
}

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
    attachTestCamera(state);
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
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBeCloseTo(20, 7);
        expect(state.of(GlobalTransform).scale.x.get(eid)).toBeCloseTo(21, 7);
        state.step(Time.FIXED_DT * 0.25);
        expect(state.time.fixedSteps).toBe(0);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBeCloseTo(20, 7);
        words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[row * 12]).toBeCloseTo(17.5, 5);
        expect(words[row * 12 + 8]).toBeCloseTo(18.5, 5);
        state.step(Time.FIXED_DT * 0.5);
        expect(state.time.fixedSteps).toBe(1);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBeCloseTo(30, 7);
        words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[row * 12]).toBeCloseTo(22.5, 5);
        expect(() => state.globalTransformRuntime!.previous!.bytes).toThrow("GPU-only");
        expect(await bounded(state.gpu.device.popErrorScope())).toBeNull();
    } finally {
        app.dispose();
    }
});

test("a producer spawned during catch-up keeps motion after its spawn tick", async () => {
    let eid = -1;
    const app = await build({
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "CatchupSpawn",
                systems: [
                    {
                        group: "fixed",
                        update(state) {
                            if (eid < 0) {
                                eid = state.create();
                                state.add(eid, Transform);
                            }
                            Transform.pos.x.set(eid, state.time.fixedTick * 10);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const { state } = app;
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT * 2.5);
        expect(state.time.fixedTick).toBe(2);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(20);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(15, 5);
    } finally {
        app.dispose();
    }
});

test("a teleport on the first catch-up tick keeps later tick motion", async () => {
    let eid = -1;
    const app = await build({
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "CatchupTeleport",
                systems: [
                    {
                        group: "fixed",
                        update(state) {
                            if (state.time.fixedTick === 1) {
                                Transform.pos.x.set(eid, 100);
                                state.teleport(eid);
                            } else {
                                Transform.pos.x.set(eid, 100 + (state.time.fixedTick - 1) * 10);
                            }
                        },
                    },
                ],
            },
        ],
        setup(state) {
            eid = state.create();
            state.add(eid, Transform);
            Transform.pos.x.set(eid, 0);
        },
    });
    try {
        const { state } = app;
        attachTestCamera(state);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT * 2.5);
        expect(state.time.fixedTick).toBe(2);
        expect(state.time.fixedAlpha).toBeCloseTo(0.5, 5);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(110);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[table.rowIndex(eid) * 12]).toBeCloseTo(105, 5);
    } finally {
        app.dispose();
    }
});

test("a renderer interpolates GlobalTransform when the scene has no lights", async () => {
    let eid = -1;
    const app = await build({
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "FastPlacement",
                systems: [
                    {
                        group: "fixed",
                        update(state) {
                            if (eid >= 0) Transform.pos.x.set(eid, state.time.fixedTick * 4);
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
    try {
        const { state } = app;
        attachTestCamera(state);
        expect([...state.query([PointLight])]).toHaveLength(0);
        const table = globalTransformTable(state);
        state.step(Time.FIXED_DT);
        state.step(Time.FIXED_DT * 1.5);
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(words[row * 12]).toBeCloseTo(6, 5);
    } finally {
        app.dispose();
    }
});
