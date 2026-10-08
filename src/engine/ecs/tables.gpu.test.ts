import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import type { createApp, Plugin } from "../app";
import { f32, u32 } from "../index";
import { probeBuffer } from "../runtime";
import { type Component, component } from "./component";
import type { World } from "./world";

setDefaultTimeout(CEILING.gpu);

type App = Awaited<ReturnType<typeof createApp>>;
interface Subject {
    app: App;
    world: World;
    table: ReturnType<World["table"]>;
}

const configs: Parameters<typeof createApp>[0][] = [];
const apps = gpuApps(import.meta.path, configs);

/** Declare one independent world, built in the file's beforeAll; its table is declared at initialize. */
function subject(
    name: string,
    components: Component[],
    declare?: (world: World) => ReturnType<World["table"]>,
): () => Subject {
    const index = configs.length;
    const current = {} as Subject;
    const plugin: Plugin = {
        name,
        components,
        initialize(world) {
            current.world = world;
            if (declare) current.table = declare(world);
        },
    };
    configs.push({ defaults: false, plugins: [plugin] });
    return () => {
        current.app = apps()[index];
        return current;
    };
}

function bounded<T>(label: string, promise: PromiseLike<T>, timeout = 750): Promise<T> {
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

async function stepAndValidate(world: World, label: string): Promise<void> {
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    world.step(0);
    await bounded(`${label} queue completion`, device.queue.onSubmittedWorkDone());
    const error = await bounded(`${label} validation scope`, device.popErrorScope());
    if (error) throw new Error(`${label}: ${error.message}`);
}

const Record = d.struct({ amount: d.f32, tag: d.u32 });
const Rows = { amount: f32, tag: u32 };

for (const range of ["unchanged", "sparse", "partial", "full"] as const) {
    const upload = subject("TableUploadProbe", [component("Rows", Rows)], (world) =>
        world.table("table-upload-probe", Record),
    );
    test(`dense tables upload ${range} ranges with writeBuffer and skip unchanged rows`, async () => {
        const { world, table } = upload();

        const count = range === "full" ? 1000 : 100;
        const eids = Array.from({ length: count }, () => world.create());
        const slots = eids.map((eid) => table.acquire(eid));
        expect(table.eidToRowBuffer).toBeUndefined();
        await stepAndValidate(world, "unchanged table upload");
        expect(table.lastUploadPath).toBe("none");
        if (range === "unchanged") return;

        const columns = world.storage(Rows);
        const tableView = new DataView(table.bytes.buffer);
        const write = (eid: number, value: number) => {
            columns.amount.set(eid, value);
            columns.tag.set(eid, eid);
            const offset = slots[eid - eids[0]] * table.rowBytes;
            tableView.setFloat32(offset, columns.amount.get(eid), true);
            tableView.setUint32(offset + 4, columns.tag.get(eid), true);
        };
        if (range === "sparse") {
            write(eids[0], 17.25);
            table.markRange(slots[0], 1);
            await stepAndValidate(world, "sparse record range upload");
            expect(table.lastUploadPath).toBe("writeBuffer");
            const sparseRows = await bounded(
                "probe sparsely changed table row",
                probeBuffer(world, table.buffer, {
                    offset: slots[0] * table.rowBytes,
                    size: table.rowBytes,
                    label: "table-range-write-proof",
                }),
            );
            const sparseData = new DataView(sparseRows.bytes);
            expect(sparseData.getFloat32(0, true)).toBe(17.25);
            expect(sparseData.getUint32(4, true)).toBe(eids[0]);
            return;
        }

        if (range === "partial") {
            for (let i = 1; i < 100; i++) write(eids[i], i + 0.5);
            table.markRange(slots[1], 99);
            await stepAndValidate(world, "partial record range upload");
            expect(table.lastUploadPath).toBe("writeBuffer");
            return;
        }

        for (let i = 0; i < eids.length; i++) write(eids[i], i + 1000);
        table.markRange(slots[0], eids.length);
        await stepAndValidate(world, "full table range upload");
        expect(table.lastUploadPath).toBe("writeBuffer");
        const readRows = await bounded(
            "probe fully uploaded table rows",
            probeBuffer(world, table.buffer, {
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
    });
}

const Core = { x: f32 };
const Optional = { y: f32 };
const Flag = {};
const presence = subject(
    "TablePresenceProbe",
    [component("Core", Core), component("Optional", Optional), component("Flag", Flag)],
    (world) => {
        const table = world.table(
            "table-presence-probe",
            d.struct({ x: d.f32, y: d.f32, flags: d.u32 }),
        );
        table.bindComponent(Core, { x: "x" });
        table.bindFields(Optional, { y: "y" });
        table.bindMembership(Optional);
        table.bindPresence(Optional, "flags", 1);
        table.bindPresence(Flag, "flags", 2);
        return table;
    },
);

test("tables combine source fields, optional presence, and several row owners", async () => {
    const { world, table } = presence();
    const eid = world.create();
    world.add(eid, Core);
    world.add(eid, Optional);
    world.add(eid, Flag);
    world.storage(Core).x.set(eid, 4.5);
    world.storage(Optional).y.set(eid, 8.25);
    world.step(0);
    const row = table.rowIndex(eid);
    expect(row).toBeGreaterThanOrEqual(0);
    expect(table.count).toBe(1);
    const bytes = new DataView(table.bytes.buffer);
    expect(bytes.getFloat32(row * table.rowBytes, true)).toBe(4.5);
    expect(bytes.getFloat32(row * table.rowBytes + 4, true)).toBe(8.25);
    expect(bytes.getUint32(row * table.rowBytes + 8, true)).toBe(3);

    world.remove(eid, Optional);
    expect(table.rowIndex(eid)).toBe(row);
    expect(table.count).toBe(1);
    expect(bytes.getUint32(row * table.rowBytes + 8, true)).toBe(2);
    world.remove(eid, Flag);
    expect(bytes.getUint32(row * table.rowBytes + 8, true)).toBe(0);
    world.remove(eid, Core);
    expect(table.rowIndex(eid)).toBe(-1);
    expect(table.count).toBe(0);
});

const Bound = { x: f32, y: f32 };
const bound = subject("BoundTableProbe", [component("Bound", Bound)], (world) => {
    const table = world.table("bound-table-probe", d.struct({ x: d.f32, y: d.f32 }));
    table.bindComponent(Bound, { x: "x", y: "y" });
    return table;
});

test("component fields bulk-upload through a dense struct table and release their slots", async () => {
    const { world, table } = bound();

    const eid = world.create();
    world.add(eid, Bound);
    const columns = world.storage(Bound);
    columns.x.set(eid, 12.5);
    columns.y.set(eid, -4.25);
    expect(table.count).toBe(1);
    await stepAndValidate(world, "component-bound struct upload");
    const record = await bounded(
        "probe component-bound row",
        probeBuffer(world, table.buffer, {
            size: table.rowBytes,
            label: "component-bound-table-proof",
        }),
    );
    const data = new DataView(record.bytes);
    expect(data.getFloat32(0, true)).toBe(12.5);
    expect(data.getFloat32(4, true)).toBe(-4.25);

    world.remove(eid, Bound);
    expect(table.count).toBe(0);
    world.add(eid, Bound);
    expect(table.count).toBe(1);
    columns.x.set(eid, 7.5);
    await stepAndValidate(world, "reused component-bound row upload");
    const reused = await bounded(
        "probe reused component-bound row",
        probeBuffer(world, table.buffer, {
            size: table.rowBytes,
            label: "component-bound-table-reuse-proof",
        }),
    );
    expect(new DataView(reused.bytes).getFloat32(0, true)).toBe(7.5);
    world.destroy(eid);
    expect(table.count).toBe(0);
});

const tickUpload = subject("TickUploadProbe", [Bound], (world) => {
    const table = world.table("tick-upload", d.struct({ x: d.f32 }));
    table.bindComponent(Bound, { x: "x" });
    return table;
});

test("tick field writes survive until the next draw upload without advancing GPU frames", async () => {
    const { world, table } = tickUpload();
    const eid = world.create();
    world.add(eid, Bound);
    await stepAndValidate(world, "initial tick table upload");
    const frame = world.gpu.frame;
    const x = world.storage(Bound).x;
    const system = { group: "fixed" as const, update: () => x.set(eid, 42) };
    world.addSystem(system);
    world.tick();
    expect(world.gpu.frame).toBe(frame);
    world.removeSystem(system);
    await stepAndValidate(world, "tick table upload");
    const record = await bounded(
        "tick row",
        probeBuffer(world, table.buffer, { size: table.rowBytes }),
    );
    expect(new DataView(record.bytes).getFloat32(0, true)).toBe(42);
});

const dense = subject("DenseTableProbe", [], (world) =>
    world.table("dense-table-probe", d.struct({ value: d.u32 })),
);

test("dense tables reuse free-list slots, lazily publish eid mappings, and expose active rows", async () => {
    const { world, table } = dense();

    let recordBinds = 0;
    let mapBinds = 0;
    let activeBinds = 0;
    expect(table.eidToRowBuffer).toBeUndefined();
    table.subscribe(() => recordBinds++);
    table.subscribeMap(() => mapBinds++);
    table.subscribeActiveRows(() => activeBinds++);
    const first = world.create();
    const second = world.create();
    const third = world.create();
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
    await stepAndValidate(world, "initial dense-table upload");
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
    await stepAndValidate(world, "dense-table free-list reuse");

    const map = await bounded(
        "probe eid-to-row map",
        probeBuffer(world, table.eidToRowBuffer!, {
            offset: second * 4,
            size: 8,
            label: "dense-table-eid-map-proof",
        }),
    );
    expect(Array.from(new Uint32Array(map.bytes))).toEqual([secondRow + 1, reusedRow + 1]);
    const active = await bounded(
        "probe compact active-row list",
        probeBuffer(world, table.activeRowsBuffer!, {
            size: table.count * 8,
            label: "dense-table-active-rows-proof",
        }),
    );
    expect(Array.from(new Uint32Array(active.bytes))).toEqual([
        second,
        secondRow,
        third,
        reusedRow,
    ]);
    const data = await bounded(
        "probe dense table records",
        probeBuffer(world, table.buffer, {
            size: table.capacity * table.rowBytes,
            label: "dense-table-records-proof",
        }),
    );
    expect(Array.from(new Uint32Array(data.bytes))).toEqual([333, 222]);
});

const steppedGrowth = subject("SteppedTableGrowth", [], (world) =>
    world.table("stepped-table-growth", Record),
);
test("a non-placement table defers growth during a step and uploads after its copies", async () => {
    const { world, table } = steppedGrowth();
    const eid = world.create();
    table.acquire(eid);
    new DataView(table.bytes.buffer).setFloat32(0, 7, true);
    table.markRange(0, 1);
    table.upload();
    const initial = table.buffer;
    let submissions = 0;
    const queue = world.gpu.device.queue;
    const submit = queue.submit.bind(queue);
    const descriptor = Object.getOwnPropertyDescriptor(queue, "submit");
    Object.defineProperty(queue, "submit", {
        configurable: true,
        value: (buffers: GPUCommandBuffer[]) => {
            submissions++;
            submit(buffers);
        },
    });
    world.addSystem({
        group: "fixed",
        update() {
            table.reserveSlots(table.capacity + 1);
            expect(submissions).toBe(0);
            expect(world.owns(initial)).toBe(true);
            new DataView(table.bytes.buffer).setFloat32(0, 19, true);
            table.markRange(0, 1);
            table.upload();
        },
    });
    world.addSystem({
        group: "draw",
        update() {
            const encoder = world.gpu.device.createCommandEncoder();
            world.beginGpuFrame(encoder);
            queue.submit([encoder.finish()]);
            world.endGpuFrame();
        },
    });
    try {
        world.step(1 / 60);
        expect(submissions).toBe(1);
        const result = await bounded(
            "stepped growth",
            probeBuffer(world, table.buffer, { size: table.rowBytes }),
        );
        expect(new DataView(result.bytes).getFloat32(0, true)).toBe(19);
        expect(world.owns(initial)).toBe(false);
    } finally {
        if (descriptor) Object.defineProperty(queue, "submit", descriptor);
        else Reflect.deleteProperty(queue, "submit");
    }
});

const growth = subject("TableGrowthProbe", [], (world) =>
    world.table("table-growth-probe", Record),
);

test("table growth changes generation and refuses beyond the named device limit", async () => {
    const { app, table } = growth();

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
        `maxStorageBufferBindingSize (${app.world.gpu.device.limits.maxStorageBufferBindingSize} bytes)`,
    );
});

const Changed = { sparse: f32, uploaded: f32 };
const changeMarks = subject("WorldChangeMarkProbe", [component("Changed", Changed)]);

test("frame change marks clear at the world upload point", async () => {
    const { world } = changeMarks();
    const eid = world.create();
    world.add(eid, Changed);
    const storage = world.storage(Changed);
    storage.sparse.set(eid, 1);
    storage.uploaded.set(eid, 2);

    world.step(0);
    expect(
        ["sparse", "uploaded"].map((name) =>
            Array.from(world.fieldStorage(Changed, name).dirty).some((word) => word !== 0),
        ),
    ).toEqual([false, false]);

    let writeAfterUpload = false;
    const lateWriter = {
        group: "draw" as const,
        update(current: World) {
            if (writeAfterUpload) current.storage(Changed).uploaded.set(eid, 9);
        },
    };
    world.addSystem(lateWriter);
    writeAfterUpload = true;
    world.step(0);
    expect(world.fieldStorage(Changed, "uploaded").dirty[0]).not.toBe(0);

    writeAfterUpload = false;
    world.step(0);
    expect(world.fieldStorage(Changed, "uploaded").dirty[0]).toBe(0);
});
