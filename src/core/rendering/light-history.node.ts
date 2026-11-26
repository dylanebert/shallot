import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { probeBuffer, Time } from "../../engine";
import { GlobalTransform, resetInterpolation, Transform } from "../transform";
import {
    GlobalTransformHistory,
    globalTransformTable,
    readRenderGlobalTransform,
} from "./global-transform";
import { attachTexture, Camera, RenderingPlugin } from "./index";
import { PointLight, SpotLight } from "./lighting";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

test("unread new and reattached lights snap until fixed capture; read ones match GPU history, as all do across ticks and recovery", async () => {
    let x = 0;
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "LightHistoryWriter",
                recovery: () => ({
                    snapshot: () => x,
                    restore: (value) => {
                        x = value as number;
                    },
                }),
            },
        ],
    });
    const { world } = app;
    try {
        const camera = world.create();
        world.add(camera, Camera);
        world.add(camera, Transform);
        attachTexture(world, camera, { width: 8, height: 8 });
        const table = globalTransformTable(world);
        // Never read before their lights attach, so their lights start without history.
        const eids = [world.create(), world.create()];
        // Read before their lights attach, which enrolls them.
        const readEids = [world.create(), world.create()];
        const placed = [...eids, ...readEids];
        for (const eid of placed) world.add(eid, Transform);
        const out = new Float32Array(12);
        world.addSystem(
            {
                group: "fixed",
                update(world) {
                    for (const eid of placed) {
                        const storage = world.storage(Transform);
                        storage.translation.set(eid, x, x * 2, -x, 0);
                        storage.rotation.set(eid, 0, Math.sin(x), 0, Math.cos(x));
                        storage.scale.set(eid, 1 + x / 100, 2, 3, 0);
                    }
                },
            },
            "LightHistoryWriter",
        );
        world.step(Time.FIXED_DT);
        for (const eid of readEids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).toBe(0);
        }
        let attach = true;
        world.addSystem({
            group: "simulation",
            update(world) {
                if (!attach) return;
                attach = false;
                for (const eid of [eids[0]!, readEids[0]!]) world.add(eid, PointLight);
                for (const eid of [eids[1]!, readEids[1]!]) world.add(eid, SpotLight);
            },
        });
        async function gpuRows() {
            const result = await probeBuffer(world, table.buffer, { size: table.buffer.size });
            return new Float32Array(result.bytes);
        }
        async function matchesGpu(subjects = placed) {
            const gpu = await gpuRows();
            for (const eid of subjects) {
                readRenderGlobalTransform(world, eid, out);
                for (const lane of [0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
                    expect(out[lane]).toBeCloseTo(gpu[table.rowIndex(eid) * 12 + lane]!, 5);
            }
        }
        x = 20;
        world.step(Time.FIXED_DT * 1.5);
        for (const eid of eids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).toBe(20);
        }
        const gpu = await gpuRows();
        expect(gpu[table.rowIndex(eids[0]!) * 12]).toBeCloseTo(10, 5);
        for (const eid of readEids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).toBeCloseTo(10, 5);
        }
        await matchesGpu(readEids);
        x = 30;
        world.step(Time.FIXED_DT);
        await matchesGpu();
        x = 40;
        world.step(Time.FIXED_DT * 2);
        await matchesGpu();
        world.step(0);
        await matchesGpu();
        const snapshot = world.snapshot();
        x = 50;
        world.step(Time.FIXED_DT);
        await matchesGpu();
        world.restore(snapshot);
        world.step(0);
        await matchesGpu();
        x = 60;
        world.tick();
        world.tick();
        world.step(0);
        await matchesGpu();
        for (const eid of eids) {
            world.storage(Transform).translation.x.set(eid, 100);
            resetInterpolation(world, eid);
        }
        world.step(0);
        await matchesGpu();
        for (const eid of [eids[0]!, readEids[0]!]) world.remove(eid, PointLight);
        for (const eid of [eids[1]!, readEids[1]!]) world.remove(eid, SpotLight);
        x = 70;
        world.step(Time.FIXED_DT);
        attach = true;
        x = 80;
        world.step(Time.FIXED_DT);
        for (const eid of eids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).toBe(80);
        }
        for (const eid of readEids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).not.toBe(80);
        }
        await matchesGpu(readEids);
        x = 90;
        world.step(Time.FIXED_DT);
        await matchesGpu();
        const runtime = world.resource(GlobalTransformHistory);
        const generation = table.generation;
        const spawned: number[] = [];
        for (let i = 0; i < 130; i++) {
            const eid = world.create();
            spawned.push(eid);
            world.add(eid, Transform);
            world.storage(Transform).translation.x.set(eid, i);
        }
        world.step(0);
        expect(table.generation).toBeGreaterThan(generation);
        for (const eid of spawned.slice(0, 65)) world.destroy(eid);
        for (let i = 0; i < 65; i++) world.add(world.create(), Transform);
        world.step(0);
        await matchesGpu();
        for (const follower of [runtime.previous!, table]) {
            expect(follower.activeRowsBuffer).toBe(runtime.current!.activeRowsBuffer);
            expect(follower.eidToRowBuffer).toBe(runtime.current!.eidToRowBuffer);
            expect(follower.activeGeneration).toBe(runtime.current!.activeGeneration);
            expect(follower.mapGeneration).toBe(runtime.current!.mapGeneration);
            expect(follower.count).toBe(runtime.current!.count);
            for (const eid of world.query([GlobalTransform]))
                expect(follower.rowIndex(eid)).toBe(runtime.current!.rowIndex(eid));
        }
        expect(world.storage(GlobalTransform).translation.x.get(eids[0]!)).toBe(90);
    } finally {
        await disposeGpuApps([app]);
    }
});
