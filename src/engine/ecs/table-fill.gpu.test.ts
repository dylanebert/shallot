import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { f32, vec2 } from "../index";
import { probeBuffer } from "../runtime";
import { registration } from "./traits";

setDefaultTimeout(CEILING.gpu);
const Rows = { amount: f32, vector: vec2 };
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [{ name: "BoundFill", components: [registration("Rows", Rows)] }],
    },
]);

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
    const app = subjects()[0];
    const world = app.world;
    try {
        const table = world.table("bound-fill", d.struct({ amount: d.f32 }));
        table.bindComponent(Rows, { amount: "amount" });
        const a = world.create();
        const b = world.create();
        world.add(a, Rows);
        world.add(b, Rows);
        const rowA = table.acquire(a);
        const rowB = table.acquire(b);
        const columns = world.storage(Rows);
        columns.amount.set(a, 3);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(0);
        world.step(0);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(3);
        const scalarBytes = new Float32Array(
            (
                await bounded(
                    "scalar set",
                    probeBuffer(world, table.buffer, { size: table.buffer.size }),
                )
            ).bytes,
        );
        expect(scalarBytes[rowA]).toBe(3);
        columns.amount.writeEncoded(new Uint32Array([a, b]), new Float32Array([7, 11]));
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(3);
        world.step(0);
        expect(new DataView(table.bytes.buffer).getFloat32(rowA * 4, true)).toBe(7);
        expect(new DataView(table.bytes.buffer).getFloat32(rowB * 4, true)).toBe(11);
        const bulkBytes = new Float32Array(
            (
                await bounded(
                    "bulk write",
                    probeBuffer(world, table.buffer, { size: table.buffer.size }),
                )
            ).bytes,
        );
        expect([bulkBytes[rowA], bulkBytes[rowB]]).toEqual([7, 11]);
        const vectorTable = world.table("bound-vector-fill", d.struct({ value: d.vec2f }));
        vectorTable.bindComponent(Rows, { value: "vector" });
        world.step(0); // Seed membership before the unmarked control.
        const vectorRow = vectorTable.rowIndex(b);
        columns.vector.y.set(b, 13);
        world.step(0);
        const laneBytes = new Float32Array(
            (
                await bounded(
                    "lane set",
                    probeBuffer(world, vectorTable.buffer, { size: vectorTable.buffer.size }),
                )
            ).bytes,
        );
        expect(Array.from(laneBytes.subarray(vectorRow * 2, vectorRow * 2 + 2))).toEqual([0, 13]);
        columns.vector.y.writeEncoded(new Uint32Array([b]), new Float32Array([31]));
        world.step(0);
        const laneBulkBytes = new Float32Array(
            (
                await bounded(
                    "lane bulk write",
                    probeBuffer(world, vectorTable.buffer, { size: vectorTable.buffer.size }),
                )
            ).bytes,
        );
        expect(Array.from(laneBulkBytes.subarray(vectorRow * 2, vectorRow * 2 + 2))).toEqual([
            0, 31,
        ]);
        columns.amount.column[a] = 17;
        columns.amount.markChanged(a);
        columns.amount.column[b] = 19; // Deliberately not published.
        columns.vector.column[b * 2] = 23;
        columns.vector.x.markChanged(b);
        world.step(0);
        const marked = new Float32Array(
            (
                await bounded(
                    "marked paths",
                    probeBuffer(world, table.buffer, { size: table.buffer.size }),
                )
            ).bytes,
        );
        expect(marked[rowA]).toBe(17);
        expect(marked[rowB]).toBe(11);
        const vectorBytes = new Float32Array(
            (
                await bounded(
                    "lane paths",
                    probeBuffer(world, vectorTable.buffer, { size: vectorTable.buffer.size }),
                )
            ).bytes,
        );
        expect(Array.from(vectorBytes.subarray(vectorRow * 2, vectorRow * 2 + 2))).toEqual([
            23, 31,
        ]);
        world.remove(a, Rows);
        world.step(0);
        expect(table.rowIndex(a)).toBe(-1);
        expect(table.count).toBe(1);
        // A non-component producer can reuse the freed row without bound columns overwriting it.
        const c = world.create();
        const rowC = table.acquire(c);
        expect(rowC).toBe(rowA);
        columns.amount.set(c, 99);
        new DataView(table.bytes.buffer).setFloat32(rowC * 4, 29, true);
        table.markRange(rowC, 1);
        world.step(0);
        const result = await bounded(
            "bound table readback",
            probeBuffer(world, table.buffer, { size: table.buffer.size }),
        );
        const values = new Float32Array(result.bytes);
        expect(values[rowB]).toBe(11);
        expect(values[rowC]).toBe(29);
    } finally {
        app.dispose();
    }
});
