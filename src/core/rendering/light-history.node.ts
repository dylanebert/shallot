import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createTestGpuApp } from "../../../scripts/gpu.fixture";
import { probeBuffer, Time } from "../../engine";
import { GlobalTransform, resetInterpolation, Transform } from "../transform";
import { globalTransformTable, readRenderGlobalTransform } from "./global-transform";
import { attachTexture, Camera, RenderingPlugin } from "./index";
import { PointLight, SpotLight } from "./lighting";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

test("new and reattached lights snap until fixed capture, then match GPU history across ticks and recovery", async () => {
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
        const eids = [world.create(), world.create()];
        for (const eid of eids) world.add(eid, Transform);
        const out = new Float32Array(12);
        world.addSystem(
            {
                group: "fixed",
                update(world) {
                    for (const eid of eids) {
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
        expect(() => readRenderGlobalTransform(world, eids[0]!, out)).toThrow(
            "PointLight or SpotLight",
        );
        let attach = true;
        world.addSystem({
            group: "simulation",
            update(world) {
                if (!attach) return;
                attach = false;
                world.add(eids[0]!, PointLight);
                world.add(eids[1]!, SpotLight);
            },
        });
        async function matchesGpu() {
            const result = await probeBuffer(world, table.buffer, { size: table.buffer.size });
            const gpu = new Float32Array(result.bytes);
            for (const eid of eids) {
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
        const gpu = new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        );
        expect(gpu[table.rowIndex(eids[0]!) * 12]).toBeCloseTo(10, 5);
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
        world.remove(eids[0]!, PointLight);
        world.remove(eids[1]!, SpotLight);
        x = 70;
        world.step(Time.FIXED_DT);
        attach = true;
        x = 80;
        world.step(Time.FIXED_DT);
        for (const eid of eids) {
            readRenderGlobalTransform(world, eid, out);
            expect(out[0]).toBe(80);
        }
        x = 90;
        world.step(Time.FIXED_DT);
        await matchesGpu();
        expect(world.storage(GlobalTransform).translation.x.get(eids[0]!)).toBe(90);
    } finally {
        app.dispose();
    }
});
