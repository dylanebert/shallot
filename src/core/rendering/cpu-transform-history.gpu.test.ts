import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { Time } from "../../engine";
import { GlobalTransform, Transform } from "../transform";
import {
    GlobalTransformHistory,
    RenderTransformHistory,
    readRenderGlobalTransform,
    retainRenderTransformHistory,
} from "./global-transform";
import { Camera, PointLight, RenderingPlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
let target = -1;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "Motion",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            world
                                .storage(Transform)
                                .translation.set(target, world.time.fixedTick * 10, 0, 0);
                        },
                    },
                ],
            },
        ],
        setup(world) {
            for (let i = 0; i < 1024; i++) {
                const eid = world.create();
                world.add(eid, Transform);
                if (i === 0) target = eid;
            }
        },
    },
]);

test("1024 GPU placements without CPU readers or corrections retain no CPU history", () => {
    const world = subjects()[0].world;
    world.step(Time.FIXED_DT * 2.5);
    const runtime = world.resource(GlobalTransformHistory);
    expect(runtime.current!.count).toBe(1024);
    expect(runtime.cpuHistory.size).toBe(0);
    expect(runtime.cpuReaders.length).toBe(0);
});

test("explicit enrollment captures interpolation before the first read and removal releases its samples", () => {
    const world = subjects()[0].world;
    world.add(target, RenderTransformHistory);
    world.tick();
    world.step(0);
    const out = new Float32Array(12);
    readRenderGlobalTransform(world, target, out);
    expect(out[0]).toBeCloseTo((world.time.fixedTick - 1 + world.time.fixedAlpha) * 10, 5);
    const runtime = world.resource(GlobalTransformHistory);
    expect(runtime.cpuHistory.size).toBe(1);
    expect(runtime.cpuHistory.get(target)!.previous.byteLength).toBe(48);
    expect(runtime.cpuHistory.get(target)!.current.byteLength).toBe(48);
    world.remove(target, RenderTransformHistory);
    expect(runtime.cpuHistory.size).toBe(0);
    expect(runtime.cpuReaders.length).toBe(0);
});

test("leases, explicit enrollment and placement lifetimes remain independent", () => {
    const world = subjects()[0].world;
    const eid = world.create();
    world.add(eid, Transform);
    const history = world.resource(GlobalTransformHistory).cpuHistory;
    const a = retainRenderTransformHistory(world, eid);
    const b = retainRenderTransformHistory(world, eid);
    const samples = history.get(eid);
    a();
    a();
    expect(history.get(eid)).toBe(samples);
    world.add(eid, RenderTransformHistory);
    b();
    expect(history.get(eid)).toBe(samples);
    const c = retainRenderTransformHistory(world, eid);
    world.remove(eid, RenderTransformHistory);
    expect(history.get(eid)).toBe(samples);
    c();
    expect(history.has(eid)).toBe(false);
    const stale = retainRenderTransformHistory(world, eid);
    world.remove(eid, GlobalTransform);
    expect(history.has(eid)).toBe(false);
    world.add(eid, GlobalTransform);
    const fresh = retainRenderTransformHistory(world, eid);
    stale();
    expect(history.has(eid)).toBe(true);
    fresh();
    expect(history.has(eid)).toBe(false);
    world.destroy(eid);
});

test("automatic camera and light requests retire independently without a shared marker", () => {
    const world = subjects()[0].world;
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.tick();
    const runtime = world.resource(GlobalTransformHistory);
    const samples = runtime.cpuHistory.get(eid);
    expect(samples!.captured).toBe(true);
    world.add(eid, PointLight);
    expect(runtime.cpuHistory.get(eid)).toBe(samples);
    expect(samples!.captured).toBe(true);
    expect(world.has(eid, RenderTransformHistory)).toBe(false);
    expect(runtime.cpuHistory.has(eid)).toBe(true);
    world.remove(eid, Camera);
    expect(runtime.cpuHistory.has(eid)).toBe(true);
    world.remove(eid, PointLight);
    expect(runtime.cpuHistory.has(eid)).toBe(false);
    world.add(eid, Camera);
    world.remove(eid, GlobalTransform);
    expect(runtime.cpuHistory.has(eid)).toBe(false);
    expect(runtime.cpuRequests.has(eid)).toBe(false);
    world.add(eid, GlobalTransform);
    expect(runtime.cpuHistory.has(eid)).toBe(true);
    world.destroy(eid);
    expect(runtime.cpuHistory.has(eid)).toBe(false);
    expect(runtime.cpuRequests.has(eid)).toBe(false);
});

test("cleanup", async () => {
    await disposeGpuApps(subjects());
});
