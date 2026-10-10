import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { probeBuffer } from "./probe";

setDefaultTimeout(CEILING.gpu);
const kinds = ["records", "active", "map"] as const;
const subjects = gpuApps(
    import.meta.path,
    kinds.map(() => ({ defaults: false, plugins: [] })),
);

for (const [index, kind] of kinds.entries()) {
    test(`one-shot readback reads a table's grown ${kind} buffer without mutating a prior result`, async () => {
        const app = subjects()[index];
        const world = app.world;
        const table = world.table(`readback-${kind}`, d.struct({ value: d.u32 }));
        const row = table.acquire(4);
        if (kind === "map") table.enableEidLookup();
        new Uint32Array(table.bytes.buffer)[row] = 42;
        table.markRange(row, 1);
        table.upload();
        const source = () =>
            kind === "records"
                ? table.buffer
                : kind === "active"
                  ? table.activeRowsBuffer!
                  : table.eidToRowBuffer!;
        try {
            const original = await probeBuffer(world, source());
            const oldBytes = original.bytes.slice(0);
            const second = table.acquire(33);
            new Uint32Array(table.bytes.buffer)[second] = 73;
            table.markRange(second, 1);
            table.upload();
            expect(source().size).toBeGreaterThan(original.bytes.byteLength);
            world.gpu.frame = 9;
            const current = await probeBuffer(world, source());
            expect(current.frame).toBe(9);
            expect(current.bytes.byteLength).toBe(source().size);
            expect(new Uint8Array(original.bytes)).toEqual(new Uint8Array(oldBytes));
            const values = new Uint32Array(current.bytes);
            if (kind === "records") expect([...values]).toEqual([42, 73]);
            else if (kind === "active") expect([...values]).toEqual([4, 0, 33, 1]);
            else {
                expect(values[4]).toBe(1);
                expect(values[33]).toBe(2);
            }
        } finally {
            await disposeGpuApps([app]);
        }
    });
}
