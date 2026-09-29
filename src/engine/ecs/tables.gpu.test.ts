import { afterEach, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { build, type Plugin } from "../app";
import { f32, field, u32 } from "../index";
import { probeBuffer } from "../runtime";
import type { State } from "./state";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const apps: Awaited<ReturnType<typeof build>>[] = [];
afterEach(() => {
    for (const app of apps.splice(0)) app.dispose();
});

function bounded<T>(label: string, promise: PromiseLike<T>, timeout = 5_000): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`${label} timed out after ${timeout} ms`)),
            timeout,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error: unknown) => {
                clearTimeout(timer);
                reject(new Error(`${label} rejected: ${String(error)}`, { cause: error }));
            },
        );
    });
}

async function stepAndValidate(state: State, label: string): Promise<void> {
    const device = state.gpu.device;
    device.pushErrorScope("validation");
    state.step(0);
    await bounded(`${label} queue completion`, device.queue.onSubmittedWorkDone());
    const error = await bounded(`${label} validation scope`, device.popErrorScope());
    if (error) throw new Error(`${label}: ${error.message}`);
}

const Record = d.struct({ amount: d.f32, tag: d.u32 });
const Rows = { amount: field(f32), tag: field(u32) };

test("dense tables choose no upload, mapped scatter, or one full range upload and preserve rows", async () => {
    let state!: State;
    let table!: ReturnType<State["table"]>;
    const plugin: Plugin = {
        name: "TableUploadProbe",
        components: { Rows },
        initialize(current) {
            state = current;
            table = current.table("table-upload-probe", Record, { uploadThreshold: 0.15 });
        },
    };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);

    const eids = Array.from({ length: 1000 }, () => state.create());
    const slots = eids.map((eid) => table.acquire(eid));
    expect(table.eidToRowBuffer).toBeUndefined();
    await stepAndValidate(state, "unchanged table upload");
    expect(table.lastUploadPath).toBe("none");

    const columns = state.of(Rows);
    const tableView = new DataView(table.bytes.buffer);
    const write = (eid: number, value: number) => {
        columns.amount.set(eid, value);
        columns.tag.set(eid, eid);
        const offset = slots[eid - eids[0]] * table.rowBytes;
        tableView.setFloat32(offset, columns.amount.get(eid), true);
        tableView.setUint32(offset + 4, columns.tag.get(eid), true);
    };
    write(eids[0], 17.25);
    table.markRange(slots[0], 1);
    await stepAndValidate(state, "sparse table scatter");
    expect(table.lastUploadPath).toBe("scatter");
    const sparseRows = await bounded(
        "probe scattered table rows",
        probeBuffer(state.gpu.device, table.buffer, {
            offset: slots[0] * table.rowBytes,
            size: table.rowBytes,
            label: "table-scatter-proof",
        }),
    );
    const sparseData = new DataView(sparseRows.bytes);
    expect(sparseData.getFloat32(0, true)).toBe(17.25);
    expect(sparseData.getUint32(4, true)).toBe(eids[0]);

    for (let i = 1; i < 100; i++) write(eids[i], i + 0.5);
    table.markRange(slots[1], 99);
    await stepAndValidate(state, "partial table scatter");
    expect(table.lastUploadPath).toBe("scatter");

    for (let i = 0; i < eids.length; i++) write(eids[i], i + 1000);
    table.markRange(slots[0], eids.length);
    await stepAndValidate(state, "full table range upload");
    expect(table.lastUploadPath).toBe("writeBuffer");
    const readRows = await bounded(
        "probe fully uploaded table rows",
        probeBuffer(state.gpu.device, table.buffer, {
            offset: 0,
            size: eids.length * d.sizeOf(Record),
            label: "table-full-upload-proof",
        }),
    );
    const actual = new DataView(readRows.bytes);
    for (let i = 0; i < eids.length; i++) {
        expect(actual.getFloat32(i * 8, true)).toBe(columns.amount.get(eids[i]));
        expect(actual.getUint32(i * 8 + 4, true)).toBe(columns.tag.get(eids[i]));
    }
}, 30_000);

test("dense tables reuse free-list slots, lazily publish eid mappings, and expose active rows", async () => {
    let state!: State;
    let table!: ReturnType<State["table"]>;
    const plugin: Plugin = {
        name: "DenseTableProbe",
        initialize(current) {
            state = current;
            table = current.table("dense-table-probe", d.struct({ value: d.u32 }));
        },
    };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);

    let recordBinds = 0;
    let mapBinds = 0;
    let activeBinds = 0;
    expect(table.eidToRowBuffer).toBeUndefined();
    table.subscribe(() => recordBinds++);
    table.subscribeMap(() => mapBinds++);
    table.subscribeActiveRows(() => activeBinds++);
    const first = state.create();
    const second = state.create();
    const third = state.create();
    expect(mapBinds).toBe(1);
    const firstRow = table.acquire(first);
    const secondRow = table.acquire(second);
    expect(mapBinds).toBe(3);
    expect([firstRow, secondRow]).toEqual([0, 1]);
    expect(recordBinds).toBe(2);
    expect(activeBinds).toBe(2);
    const values = new Uint32Array(table.bytes.buffer);
    values[firstRow] = 111;
    values[secondRow] = 222;
    table.markRange(firstRow, 2);
    await stepAndValidate(state, "initial dense-table upload");
    expect(table.lastUploadPath).toBe("writeBuffer");

    table.release(first);
    const reusedRow = table.acquire(third);
    expect(reusedRow).toBe(firstRow);
    expect(table.rowIndex(first)).toBe(-1);
    expect(table.rowIndex(second)).toBe(secondRow);
    expect(table.rowIndex(third)).toBe(reusedRow);
    expect([recordBinds, mapBinds, activeBinds]).toEqual([2, 3, 2]);
    values[reusedRow] = 333;
    table.markRange(reusedRow, 1);
    await stepAndValidate(state, "dense-table free-list reuse");

    const map = await bounded(
        "probe eid-to-row map",
        probeBuffer(state.gpu.device, table.eidToRowBuffer!, {
            offset: second * 4,
            size: 8,
            label: "dense-table-eid-map-proof",
        }),
    );
    expect(Array.from(new Uint32Array(map.bytes))).toEqual([secondRow + 1, reusedRow + 1]);
    const active = await bounded(
        "probe compact active-row list",
        probeBuffer(state.gpu.device, table.activeRowsBuffer!, {
            size: table.count * 4,
            label: "dense-table-active-rows-proof",
        }),
    );
    expect(Array.from(new Uint32Array(active.bytes))).toEqual([secondRow, reusedRow]);
    const data = await bounded(
        "probe dense table records",
        probeBuffer(state.gpu.device, table.buffer, {
            size: table.capacity * table.rowBytes,
            label: "dense-table-records-proof",
        }),
    );
    expect(Array.from(new Uint32Array(data.bytes))).toEqual([333, 222]);
}, 20_000);

test("table growth changes generation and refuses beyond the named device limit", async () => {
    let table!: ReturnType<State["table"]>;
    const plugin: Plugin = {
        name: "TableGrowthProbe",
        initialize(state) {
            table = state.table("table-growth-probe", Record);
        },
    };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);

    let reboundBuffer: GPUBuffer | undefined;
    let reboundGeneration = 0;
    let binds = 0;
    table.subscribe((buffer, generation) => {
        reboundBuffer = buffer;
        reboundGeneration = generation;
        binds++;
    });
    const generation = table.generation;
    table.reserveSlots(table.capacity + 1);
    expect(table.generation).toBeGreaterThan(generation);
    expect(reboundGeneration).toBe(table.generation);
    expect(reboundBuffer).toBe(table.buffer);
    expect(binds).toBe(2);
    expect(table.buffer.size).toBe(table.capacity * d.sizeOf(Record));

    expect(() => table.reserveSlots(table.maxRows + 1)).toThrow(
        `maxStorageBufferBindingSize (${app.state.gpu.device.limits.maxStorageBufferBindingSize} bytes)`,
    );
}, 20_000);
