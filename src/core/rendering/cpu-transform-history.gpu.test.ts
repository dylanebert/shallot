import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { Time } from "../../engine";
import { Transform } from "../transform";
import {
    GlobalTransformHistory,
    RenderTransformHistory,
    readRenderGlobalTransform,
} from "./global-transform";
import { RenderingPlugin } from "./index";

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

test("cleanup", async () => {
    await disposeGpuApps(subjects());
});
