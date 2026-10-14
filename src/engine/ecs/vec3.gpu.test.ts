import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { vec3, vec4 } from "../index";
import { probeBuffer } from "../runtime";
import { component, sameComponentSchema, sameTypeLayout } from "./component";
import { World } from "./world";

setDefaultTimeout(TIMEOUT.gpu);

// The same xyz data declared both ways; every observation of `xyz` must equal `wide`'s.
const Pair = component("vec3-layout-pair", { xyz: vec3, wide: vec4 });
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [{ name: "Vec3Layout", components: [Pair] }] },
]);

test("vec3 has vec4's layout, and three authored values leave the fourth lane 0", () => {
    expect(sameTypeLayout(vec3, vec4)).toBe(true);
    expect(sameComponentSchema({ v: vec3 }, { v: vec4 })).toBe(true);
    const world = new World();
    try {
        world.registry.register(Pair);
        const eid = world.create();
        world.add(eid, Pair, { xyz: [1, 2, 3], wide: [1, 2, 3, 0] });
        const pair = world.storage(Pair);
        expect(pair.xyz.column.length).toBe(pair.wide.column.length);
        expect(Array.from(pair.xyz.column.subarray(eid * 4, eid * 4 + 4))).toEqual([1, 2, 3, 0]);
        const four = world.create();
        world.add(four, Pair, { xyz: [4, 5, 6, 7] });
        expect(pair.xyz.w.get(four)).toBe(7);
    } finally {
        world.dispose();
    }
});

test("a vec3 field round-trips through a snapshot as vec4 does", () => {
    const world = new World();
    try {
        world.registerRecovery("Vec3Snapshot", "stateless");
        world.registry.register(Pair);
        const eid = world.create();
        world.add(eid, Pair, { xyz: [1.5, -2, 3.25, 9], wide: [1.5, -2, 3.25, 9] });
        const pair = world.storage(Pair);
        const saved = world.snapshot();
        pair.xyz.set(eid, 0, 0, 0, 0);
        pair.wide.set(eid, 0, 0, 0, 0);
        world.restore(saved);
        const xyz = Array.from(world.storage(Pair).xyz.column.subarray(eid * 4, eid * 4 + 4));
        const wide = Array.from(world.storage(Pair).wide.column.subarray(eid * 4, eid * 4 + 4));
        expect(wide).toEqual([1.5, -2, 3.25, 9]);
        expect(xyz).toEqual(wide);
    } finally {
        world.dispose();
    }
});

test("a vec3 field binds to vec3f and vec4f table records and uploads as vec4 does", async () => {
    const app = subjects()[0];
    const world = app.world;
    try {
        const tables = (["vec3f", "vec4f"] as const).map((kind) => {
            const type = kind === "vec3f" ? d.vec3f : d.vec4f;
            const table = world.table(`vec3-layout-${kind}`, d.struct({ xyz: type, wide: type }));
            table.bindComponent(Pair, { xyz: "xyz", wide: "wide" });
            return table;
        });
        const eid = world.create();
        world.add(eid, Pair, { xyz: [1.5, -2, 3.25, 9], wide: [1.5, -2, 3.25, 9] });
        world.step(0);
        for (const table of tables) {
            const row = table.rowIndex(eid);
            expect(row).toBeGreaterThanOrEqual(0);
            const bytes = new Float32Array(
                (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
            );
            const at = (row * table.rowBytes) / 4;
            const half = table.rowBytes / 8;
            const xyz = Array.from(bytes.subarray(at, at + half));
            const wide = Array.from(bytes.subarray(at + half, at + 2 * half));
            expect(xyz).toEqual(wide);
            expect(xyz.slice(0, 3)).toEqual([1.5, -2, 3.25]);
        }
    } finally {
        await disposeGpuApps([app]);
    }
});
