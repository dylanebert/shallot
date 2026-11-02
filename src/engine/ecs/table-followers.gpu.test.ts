import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { f32 } from "../index";
import { probeBuffer } from "../runtime";
import { component } from "./component";

setDefaultTimeout(TIMEOUT.gpu);
const Values = component("FollowerValues", { value: f32 });
const Flag = component("FollowerFlag", {});
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [{ name: "Followers", components: [Values, Flag] }] },
]);

test("CPU followers share rows through activation, removal, recycling, growth and restore", async () => {
    const app = subjects()[0];
    const world = app.world;
    try {
        const source = world.table("follower-source", d.struct({ unused: d.u32 }));
        const a = world.create();
        const b = world.create();
        world.add(a, Values, { value: 3 });
        world.add(b, Values, { value: 7 });
        world.add(a, Flag);
        const ra = source.acquire(a);
        const rb = source.acquire(b);
        const follower = world.table("cpu-follower", d.struct({ value: d.f32, flags: d.u32 }), {
            rowSource: source,
        });
        follower.bindFields(Values, { value: "value" });
        follower.bindPresence(Flag, "flags");
        const presence = world.table("presence-follower", d.struct({ flags: d.u32 }), {
            rowSource: source,
        });
        presence.bindPresence(Flag, "flags");
        const read = async () => {
            const result = await probeBuffer(world, follower.buffer, {
                size: follower.buffer.size,
            });
            return new DataView(result.bytes);
        };
        const row = (view: DataView, index: number) => [
            view.getFloat32(index * 8, true),
            view.getUint32(index * 8 + 4, true),
        ];
        const changed = (index: number) => {
            expect(follower.lastUploadOffset).toBe(index * 8);
            expect(follower.lastUploadSize).toBe(8);
        };
        world.step(0);
        expect(row(await read(), ra)).toEqual([3, 1]);
        expect(row(await read(), rb)).toEqual([7, 0]);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        world.storage(Values).value.set(b, 11);
        world.step(0);
        changed(rb);
        expect(row(await read(), rb)).toEqual([11, 0]);
        // A source-only record write does not dirty the follower.
        source.markRange(ra, 1);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        source.deactivate(b);
        world.storage(Values).value.set(b, 13);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        source.activate(b);
        world.step(0);
        changed(rb);
        expect(row(await read(), rb)).toEqual([13, 0]);
        world.remove(a, Values);
        world.remove(a, Flag);
        world.step(0);
        changed(ra);
        expect(row(await read(), ra)).toEqual([0, 0]);
        world.add(a, Values, { value: 17 });
        world.add(a, Flag);
        world.step(0);
        changed(ra);
        const snapshot = world.snapshot();
        source.release(a);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        const c = world.create();
        const rc = source.acquire(c);
        expect(rc).toBe(ra);
        world.step(0);
        changed(rc);
        expect(row(await read(), rc)).toEqual([0, 0]);
        expect(
            new Uint32Array(
                (await probeBuffer(world, presence.buffer, { size: presence.buffer.size })).bytes,
            )[rc],
        ).toBe(0);
        follower.enableEidLookup();
        const records: GPUBuffer[] = [];
        const maps: GPUBuffer[] = [];
        const activeLists: GPUBuffer[] = [];
        follower.subscribe((buffer, generation) => {
            expect(generation).toBe(follower.generation);
            records.push(buffer);
        });
        follower.subscribeMap((buffer, generation) => {
            expect(generation).toBe(source.mapGeneration);
            maps.push(buffer);
        });
        follower.subscribeActiveRows((buffer, generation) => {
            expect(generation).toBe(source.activeGeneration);
            activeLists.push(buffer);
        });
        const grown: { eid: number; index: number; value: number }[] = [];
        for (let i = 0; i < 20; i++) {
            const generation = follower.generation;
            const mapGeneration = source.mapGeneration;
            const activeGeneration = source.activeGeneration;
            const notifications = [records.length, maps.length, activeLists.length];
            const eid = world.create();
            world.add(eid, Values, { value: 20 + i });
            const index = source.acquire(eid);
            grown.push({ eid, index, value: 20 + i });
            expect(records.length).toBe(notifications[0] + follower.generation - generation);
            expect(maps.length).toBe(notifications[1] + source.mapGeneration - mapGeneration);
            expect(activeLists.length).toBe(
                notifications[2] + source.activeGeneration - activeGeneration,
            );
            expect(records.at(-1)).toBe(follower.buffer);
            expect(maps.at(-1)).toBe(follower.eidToRowBuffer);
            expect(activeLists.at(-1)).toBe(follower.activeRowsBuffer);
            expect(follower.capacity).toBe(source.capacity);
            world.step(0);
            changed(index);
            const view = await read();
            for (const entry of grown) expect(row(view, entry.index)).toEqual([entry.value, 0]);
            expect(row(view, rb)).toEqual([13, 0]);
        }
        expect(new Set(records).size).toBeGreaterThan(1);
        expect(new Set(maps).size).toBeGreaterThan(1);
        expect(new Set(activeLists).size).toBeGreaterThan(1);
        const newest = grown.at(-1)!;
        world.storage(Values).value.set(newest.eid, 77);
        world.step(0);
        changed(newest.index);
        expect(row(await read(), newest.index)).toEqual([77, 0]);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        // Restore retains table ownership; the producer reconciles its own membership.
        world.storage(Values).value.set(b, 99);
        world.remove(a, Flag);
        world.step(0);
        world.restore(snapshot);
        source.release(c);
        expect(source.acquire(a)).toBe(ra);
        world.step(0);
        expect(row(await read(), ra)).toEqual([17, 1]);
        expect(row(await read(), rb)).toEqual([13, 0]);
        world.step(0);
        expect(follower.lastUploadSize).toBe(0);
        expect(() => follower.acquire(a)).toThrow("row membership belongs");
        expect(() => follower.bindComponent(Values, {})).toThrow("row membership belongs");
        const owned = world.table("restored-source", d.struct({ unused: d.u32 }));
        owned.bindMembership(Flag);
        const restored = world.table(
            "restored-follower",
            d.struct({ value: d.f32, flags: d.u32 }),
            { rowSource: owned },
        );
        restored.bindFields(Values, { value: "value" });
        restored.bindPresence(Flag, "flags");
        world.step(0);
        const image = world.snapshot();
        world.remove(a, Flag);
        world.storage(Values).value.set(a, 101);
        world.add(b, Flag);
        world.step(0);
        world.restore(image);
        world.step(0);
        expect(restored.rowIndex(b)).toBe(-1);
        expect(restored.count).toBe(1);
        const restoredBytes = await probeBuffer(world, restored.buffer, {
            size: restored.buffer.size,
        });
        expect(row(new DataView(restoredBytes.bytes), restored.rowIndex(a))).toEqual([17, 1]);
        world.step(0);
        expect(restored.lastUploadSize).toBe(0);
    } finally {
        await disposeGpuApps([app]);
    }
});
