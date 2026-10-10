import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import type { createApp } from "../app";
import { probeBuffer } from "../runtime";
import type { World } from "./world";

setDefaultTimeout(CEILING.node);

const Record = d.struct({ amount: d.f32, tag: d.u32 });
const current = {} as { table: ReturnType<World["table"]> };
const configs: Parameters<typeof createApp>[0][] = [];
configs.push({
    defaults: false,
    plugins: [
        {
            name: "LargeExactTickUploads",
            initialize(world) {
                current.table = world.table("large-exact-tick-uploads", Record);
            },
        },
    ],
});
const apps = gpuApps(import.meta.path, configs);

test("four exact ticks upload 32/64/128/128 MiB without retaining uploads until a frame", async () => {
    const { world } = apps()[0];
    const { table } = current;
    table.acquire(world.create());
    const sizes = [32, 64, 128, 128].map((mib) => mib * 1024 * 1024);
    let tick = 0;
    world.addSystem({
        group: "fixed",
        update() {
            const size = sizes[tick++];
            table.reserveSlots(size / table.rowBytes);
            new DataView(table.bytes.buffer).setFloat32(0, tick, true);
            table.markRange(0, size / table.rowBytes);
            table.upload();
        },
    });
    const frame = world.gpu.frame;
    for (let i = 0; i < 4; i++) world.tick();
    expect(world.gpu.frame).toBe(frame);
    const result = await probeBuffer(world, table.buffer, { size: table.rowBytes });
    expect(new DataView(result.bytes).getFloat32(0, true)).toBe(4);
});
