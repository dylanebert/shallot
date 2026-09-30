import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { PartInput, RenderPlugin } from "../../core/rendering";
import { SearPlugin } from "../../standard/rendering";
import { Part, PartPlugin } from "../../transitional/part";
import { Transform } from "../index";
import { probeBuffer } from "../runtime";
import { build } from "./index";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
await (await import(peerModule)).setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} exceeded 5000 ms`)), 5000);
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

test("Part and Sear warm and compact a component-bound dense instance", async () => {
    const app = await build({ defaults: false, plugins: [RenderPlugin, PartPlugin, SearPlugin] });
    const { state } = app;
    try {
        const eid = state.create();
        state.add(eid, Transform);
        state.add(eid, Part);
        state.gpu.device.pushErrorScope("validation");
        state.step(0);
        await bounded("Part submissions", state.gpu.device.queue.onSubmittedWorkDone());
        expect(await bounded("Part validation", state.gpu.device.popErrorScope())).toBeNull();
        const packed = await probeBuffer(state, state.gpu.buffers.get("eids")!, { size: 4 });
        expect(new Uint32Array(packed.bytes)[0]).toBe(eid);
        const active = await probeBuffer(state, state.gpu.buffers.get("partInputs:active-rows")!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const recordSize = d.sizeOf(PartInput);
        const colorOffset = d.memoryLayoutOf(PartInput, (value) => value.color).offset;
        const materialOffset = d.memoryLayoutOf(PartInput, (value) => value.material).offset;
        const record = await probeBuffer(state, state.gpu.buffers.get("partInputs")!, {
            offset: row * recordSize,
            size: recordSize,
        });
        const data = new DataView(record.bytes);
        expect([0, 1, 2, 3].map((lane) => data.getFloat32(colorOffset + lane * 4, true))).toEqual([
            1, 0, 1, 1,
        ]);
        expect(
            [0, 1, 2, 3].map((lane) => data.getFloat32(materialOffset + lane * 4, true)),
        ).toEqual([0, 1, 0, 1]);
    } finally {
        app.dispose();
    }
});
