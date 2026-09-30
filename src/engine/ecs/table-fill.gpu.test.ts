import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import * as d from "typegpu/data";
import { build } from "../app";
import { f32, field } from "../index";
import { probeBuffer } from "../runtime";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

const Rows = { amount: field(f32) };

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 750 ms`)), 750);
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

test("bound tables gather marked columns at upload after set, bulk write and removal without field callbacks", async () => {
    const app = await build({
        defaults: false,
        plugins: [{ name: "BoundFill", components: { Rows } }],
    });
    const state = app.state;
    const observe = spyOn(state, "observeField");
    try {
        const table = state.table("bound-fill", d.struct({ amount: d.f32 }));
        table.bindComponent(Rows, { amount: "amount" });
        expect(observe).not.toHaveBeenCalled();
        const a = state.create();
        const b = state.create();
        state.add(a, Rows);
        state.add(b, Rows);
        const rowA = table.acquire(a);
        const rowB = table.acquire(b);
        const columns = state.of(Rows);
        columns.amount.set(a, 3);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(0);
        state.step(0);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(3);
        columns.amount.write(new Uint32Array([a, b]), new Float32Array([7, 11]));
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(3);
        state.step(0);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(7);
        expect(new DataView(table.bytes.buffer).getFloat32(rowB * 4, true)).toBe(11);
        state.remove(a, Rows);
        state.step(0);
        expect(table.rowIndex(a)).toBe(-1);
        expect(table.count).toBe(1);
        // A non-component producer can reuse the freed row without bound columns overwriting it.
        const c = state.create();
        const rowC = table.acquire(c);
        expect(rowC).toBe(rowA);
        columns.amount.set(c, 99);
        new DataView(table.bytes.buffer).setFloat32(rowC * 4, 29, true);
        table.markRange(rowC, 1);
        state.step(0);
        const result = await bounded(
            "bound table readback",
            probeBuffer(state, table.buffer, { size: table.buffer.size }),
        );
        const values = new Float32Array(result.bytes);
        expect(values[rowB]).toBe(11);
        expect(values[rowC]).toBe(29);
    } finally {
        observe.mockRestore();
        app.dispose();
    }
});
