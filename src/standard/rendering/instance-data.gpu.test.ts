import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { component, f32, type GpuTable, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import {
    instanceData,
    instanceDataTable,
    registerInstanceData,
    uploadInstanceData,
} from "./instance-data";

setDefaultTimeout(TIMEOUT.gpu);
const Source = component("SparseInstanceSource", { value: f32 });
const Value = component("SparseInstanceValue", { value: f32 });
const Marker = component("SparseInstanceMarker", {});
let source: GpuTable;
let entities: number[];
let store: ReturnType<typeof instanceDataTable>;
const apps = gpuApps(
    import.meta.path,
    [
        {
            defaults: false,
            plugins: [
                {
                    name: "SparseInstanceData",
                    initialize(world: World) {
                        source = world.table("sparse-source", d.struct({ value: d.f32 }));
                        source.bindComponent(Source, { value: "value" });
                        registerInstanceData(world, [
                            instanceData(Value, d.struct({ value: d.f32 })),
                            instanceData(Marker),
                        ]);
                        store = instanceDataTable(world, source);
                        entities = Array.from({ length: 3 }, () => {
                            const eid = world.create();
                            world.add(eid, Source);
                            return eid;
                        });
                    },
                },
            ],
        },
    ],
    { warmFrame: false },
);

async function read() {
    const { world } = apps()[0];
    uploadInstanceData(world);
    const bytes = (await probeBuffer(world, store.buffer)).bytes;
    const indices = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    return {
        index: (eid: number) => indices[source.rowIndex(eid)],
        value: (eid: number) => floats[indices[source.rowIndex(eid)] + 1],
        marker: (eid: number) => indices[indices[source.rowIndex(eid)] + 2],
    };
}

test("holder-only instance records follow membership, compact slots, mesh row recycling, growth and restore", async () => {
    const { world } = apps()[0];
    source.reserveSlots(262144);
    let result = await read();
    expect(store.buffer.size).toBe(262144 * 4);
    for (const eid of entities) expect(result.index(eid)).toBe(0);
    world.add(entities[0], Value, { value: 3 });
    world.add(entities[1], Value, { value: 7 });
    world.add(entities[1], Marker);
    const snapshot = world.snapshot();
    result = await read();
    expect(result.value(entities[0])).toBe(3);
    expect(result.value(entities[1])).toBe(7);
    expect(result.marker(entities[1])).toBe(1);
    expect(result.index(entities[2])).toBe(0);
    expect(store.buffer.size).toBe(262144 * 4 + 2 * 12);
    world.remove(entities[0], Value);
    result = await read();
    expect(result.index(entities[0])).toBe(0);
    expect(result.value(entities[1])).toBe(7);
    world.remove(entities[1], Source);
    world.remove(entities[0], Source);
    world.add(entities[2], Value, { value: 11 });
    world.add(entities[0], Source);
    result = await read();
    expect(result.index(entities[0])).toBe(0);
    expect(result.value(entities[2])).toBe(11);
    world.restore(snapshot);
    result = await read();
    expect(result.value(entities[0])).toBe(3);
    expect(result.value(entities[1])).toBe(7);
    expect(result.marker(entities[1])).toBe(1);
    expect(result.index(entities[2])).toBe(0);
    world.remove(entities[0], Value);
    world.remove(entities[1], Value);
    result = await read();
    expect(result.index(entities[0])).toBe(0);
    expect(result.index(entities[1])).toBeGreaterThan(0);
    expect(result.marker(entities[1])).toBe(1);
    world.remove(entities[1], Marker);
    result = await read();
    expect(store.buffer.size).toBe(262144 * 4);
    for (const eid of entities) expect(result.index(eid)).toBe(0);
});
