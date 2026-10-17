import { expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import {
    Camera,
    createApp,
    DirectionalLight,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { City, cityPlugin } from "../../examples/city/src/city";
import { layout } from "../../examples/city/src/layout";
import { expectError } from "../../scripts/test-console";
import { refuseReport, WORKLOAD } from "./workload";

function median(values: number[]): number {
    return values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]!;
}

await setupGlobals();
test("report the no-shadow city workload and reject a shadowed control", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("city workload requires a real adapter");
    console.info("[city-workload] adapter", adapter.info);
    const requiredLimits: Record<string, number> = {};
    for (const limit of [
        "maxStorageBuffersInVertexStage",
        "maxStorageBuffersInFragmentStage",
        "maxStorageTexturesInVertexStage",
        "maxStorageTexturesInFragmentStage",
    ] as const)
        if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
    const device = await adapter.requestDevice({
        requiredFeatures: ["indirect-first-instance", "timestamp-query"],
        requiredLimits,
    });
    device.addEventListener("uncapturederror", (event) => {
        throw new Error(event.error.message);
    });
    const querySet = device.createQuerySet({ type: "timestamp", count: 512 });
    const resolve = device.createBuffer({
        size: 4096,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
        size: 4096,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let active = false;
    let slots = 0;
    const labels: string[] = [];
    const createEncoder = device.createCommandEncoder.bind(device);
    Object.defineProperty(device, "createCommandEncoder", {
        configurable: true,
        value: (descriptor?: GPUCommandEncoderDescriptor) => {
            const encoder = createEncoder(descriptor);
            const render = encoder.beginRenderPass.bind(encoder);
            const compute = encoder.beginComputePass.bind(encoder);
            function writes(): GPUComputePassTimestampWrites {
                if (slots + 2 > 512) throw new Error("frame probe exhausted timestamp slots");
                const start = slots;
                slots += 2;
                return {
                    querySet,
                    beginningOfPassWriteIndex: start,
                    endOfPassWriteIndex: start + 1,
                };
            }
            Object.defineProperty(encoder, "beginRenderPass", {
                value: (d: GPURenderPassDescriptor) => {
                    if (active) labels.push(d.label ?? "render");
                    return render(active ? { ...d, timestampWrites: writes() } : d);
                },
            });
            Object.defineProperty(encoder, "beginComputePass", {
                value: (d?: GPUComputePassDescriptor) => {
                    if (active) labels.push(d?.label ?? "compute");
                    return compute(active ? { ...d, timestampWrites: writes() } : d);
                },
            });
            return encoder;
        },
    });

    let quantum = 0n;
    const gcd = (a: bigint, b: bigint): bigint => {
        while (b) [a, b] = [b, a % b];
        return a;
    };
    try {
        for (const { population, poses } of WORKLOAD.configurations) {
            const app = await createApp({
                device,
                adapter,
                plugins: [cityPlugin(WORKLOAD.seed, population)],
            }).catch((error: unknown) => {
                console.info(
                    `[city-workload] stopped building population=${population}: ${String(error)}`,
                );
                throw error;
            });
            try {
                const { world } = app;
                console.info("[city-workload] adapter", world.gpu.adapter);
                expect(world.gpu.adapter.class).toBe("real");
                const instances = [...world.query([MeshInstance])].length;
                expect(layout(WORKLOAD.seed, population).length).toBe(population);
                expect(world.resource(Meshes).size).toBe(WORKLOAD.kit.meshes);
                // Material row zero is the renderer's default, not a city kit material.
                expect(world.resource(Materials).table.highWater).toBe(WORKLOAD.kit.materials + 1);
                const camera = world.resource(City).camera;
                world.storage(Camera).far.set(camera, 30000);
                world.remove(camera, Orbit);
                attachTexture(world, camera, { width: WORKLOAD.width, height: WORKLOAD.height });
                const lights = world.storage(DirectionalLight);
                for (const eid of world.query([DirectionalLight]))
                    lights.shadowMapsEnabled.set(eid, Number(WORKLOAD.shadowMapsEnabled));
                for (const pose of poses) {
                    world
                        .storage(Transform)
                        .translation.set(
                            camera,
                            pose.translation[0],
                            pose.translation[1],
                            pose.translation[2],
                        );
                    world
                        .storage(Transform)
                        .rotation.set(
                            camera,
                            ...lookAtRotation(
                                pose.translation[0],
                                pose.translation[1],
                                pose.translation[2],
                                pose.target[0],
                                pose.target[1],
                                pose.target[2],
                            ),
                        );
                    for (let i = 0; i < 30; i++) world.step(Time.FIXED_DT);
                    await device.queue.onSubmittedWorkDone();
                    const capture = await captureTexture(world, camera);
                    const rgb = new Uint8Array(capture.width * capture.height * 3);
                    for (let i = 0; i < rgb.length / 3; i++)
                        rgb.set(capture.rgba.subarray(i * 4, i * 4 + 3), i * 3);
                    mkdirSync("test-results/city-workload", { recursive: true });
                    writeFileSync(
                        `test-results/city-workload/${population}-${pose.name}.ppm`,
                        Buffer.concat([
                            Buffer.from(`P6\n${capture.width} ${capture.height}\n255\n`),
                            rgb,
                        ]),
                    );
                    const cpu: number[] = [],
                        spans: number[] = [];
                    const passes = new Map<string, number[]>();
                    for (let frame = 0; frame < 60; frame++) {
                        slots = 0;
                        labels.length = 0;
                        active = true;
                        const start = performance.now();
                        try {
                            world.step(Time.FIXED_DT);
                        } finally {
                            active = false;
                        }
                        cpu.push(performance.now() - start);
                        refuseReport(
                            WORKLOAD.shadowMapsEnabled,
                            population,
                            [...world.query([MeshInstance])].length,
                            labels,
                        );
                        if (!slots) throw new Error("city workload produced no passes");
                        const encoder = createEncoder();
                        encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
                        encoder.copyBufferToBuffer(resolve, 0, readback, 0, slots * 8);
                        device.queue.submit([encoder.finish()]);
                        await readback.mapAsync(GPUMapMode.READ);
                        try {
                            const times = new BigUint64Array(readback.getMappedRange());
                            let first = times[0]!,
                                last = times[1]!;
                            const sums = new Map<string, number>();
                            for (let i = 1; i < slots; i++) {
                                const delta = times[i]! - times[i - 1]!;
                                quantum = gcd(quantum, delta < 0n ? -delta : delta);
                            }
                            for (let i = 0; i < slots; i += 2) {
                                first = times[i]! < first ? times[i]! : first;
                                last = times[i + 1]! > last ? times[i + 1]! : last;
                                const label = labels[i / 2]!;
                                sums.set(
                                    label,
                                    (sums.get(label) ?? 0) +
                                        Number(times[i + 1]! - times[i]!) / 1e6,
                                );
                            }
                            spans.push(Number(last - first) / 1e6);
                            for (const [label, sum] of sums) {
                                if (!passes.has(label)) passes.set(label, []);
                                passes.get(label)!.push(sum);
                            }
                        } finally {
                            readback.unmap();
                        }
                    }
                    refuseReport(WORKLOAD.shadowMapsEnabled, population, instances, [
                        ...passes.keys(),
                    ]);
                    console.info(
                        `[city-workload] adapter=${world.gpu.adapter.identity} observed-timestamp-quantum-ns=${quantum} (GCD of observed timestamp differences)`,
                    );
                    console.info(
                        `[city-workload] population=${population} pose=${pose.name} resolution=${WORKLOAD.width}x${WORKLOAD.height} samples=60 median-step-cpu-ms=${median(cpu).toFixed(3)} median-gpu-frame-span-ms=${median(spans).toFixed(3)} median-pass-sums-ms=${JSON.stringify(Object.fromEntries([...passes].map(([label, values]) => [label, Number(median(values).toFixed(3))])))}`,
                    );
                }
                if (population === WORKLOAD.populations[0]) {
                    for (const eid of world.query([DirectionalLight]))
                        lights.shadowMapsEnabled.set(eid, 1);
                    slots = 0;
                    labels.length = 0;
                    active = true;
                    try {
                        world.step(Time.FIXED_DT);
                    } finally {
                        active = false;
                    }
                    await device.queue.onSubmittedWorkDone();
                    expect(labels.some((label) => /shadow/i.test(label))).toBe(true);
                    expect(() => refuseReport(true, population, instances, labels)).toThrow(
                        "shadow-enabled",
                    );
                    expect(() => refuseReport(false, population, instances, labels)).toThrow(
                        "shadow pass",
                    );
                    console.info(
                        `[city-workload] shadowed control refused: enabled record and observed shadow passes (${labels.filter((label) => /shadow/i.test(label)).join(",")})`,
                    );
                }
            } catch (error) {
                console.info(
                    `[city-workload] stopped at population=${population}: ${String(error)}`,
                );
                throw error;
            } finally {
                app.dispose();
            }
        }
        console.info(
            "[city-workload] GPU frame span is first pass start to last pass end; pass timestamps omit work outside their passes. CPU measures world.step only, excluding timestamp readback.",
        );
    } finally {
        querySet.destroy();
        resolve.destroy();
        readback.destroy();
        expectError(/GPU device lost destroyed: device was destroyed/);
        device.destroy();
        await device.lost;
    }
}, 0);
