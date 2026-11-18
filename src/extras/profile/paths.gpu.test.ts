import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { DiagnosticsStore } from "../../engine";
import { ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const cases = ["render", "high indices"] as const;
const subjects = compileGpuFile(import.meta.path, async () => {
    const subjects = [];
    for (const rejection of cases) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("profiler adapter unavailable");
        const device = await adapter.requestDevice({
            requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
        });
        const invalid = device.createQuerySet({ type: "occlusion", count: 4096 });
        const createEncoder = device.createCommandEncoder.bind(device);
        device.createCommandEncoder = (desc) => {
            const encoder = createEncoder(desc);
            const render = encoder.beginRenderPass.bind(encoder);
            const compute = encoder.beginComputePass.bind(encoder);
            const writes = (writes: GPUComputePassTimestampWrites | undefined, render: boolean) =>
                writes &&
                ((rejection === "render" && render) ||
                    (rejection === "high indices" && (writes.endOfPassWriteIndex ?? 0) >= 4094))
                    ? { ...writes, querySet: invalid }
                    : writes;
            encoder.beginRenderPass = (desc) =>
                render({ ...desc, timestampWrites: writes(desc.timestampWrites, true) });
            encoder.beginComputePass = (desc) =>
                compute({ ...desc, timestampWrites: writes(desc?.timestampWrites, false) });
            return encoder;
        };
        const app = await createTestGpuApp({
            defaults: false,
            plugins: [ProfilePlugin],
            device,
            adapter,
        });
        const target = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        subjects.push({ app, device, target, invalid });
    }
    return subjects;
});

for (const [index, rejection] of cases.entries()) {
    test(`rejecting only ${rejection} timestamps leaves render and compute frames valid`, async () => {
        const { app, device, target, invalid } = subjects()[index];
        try {
            for (let frame = 0; frame < 3; frame++) {
                device.pushErrorScope("validation");
                app.world.step(1 / 60);
                const encoder = device.createCommandEncoder();
                const renderWrites = app.world.gpu.span!("render");
                expect(renderWrites).toBeUndefined();
                encoder
                    .beginRenderPass({
                        colorAttachments: [
                            { view: target.createView(), loadOp: "clear", storeOp: "store" },
                        ],
                        timestampWrites: renderWrites,
                    })
                    .end();
                for (let i = 1; i < 2047; i++) app.world.gpu.span!("unused");
                const computeWrites = app.world.gpu.span!("compute-high");
                expect(computeWrites).toBeUndefined();
                encoder.beginComputePass({ timestampWrites: computeWrites }).end();
                device.queue.submit([encoder.finish()]);
                expect(await device.popErrorScope()).toBeNull();
                expect(app.world.gpu.span!("unavailable")).toBeUndefined();
                expect(
                    app.world.resource(DiagnosticsStore).get("render/timestamp_query")?.value(),
                ).toBe(0);
            }
        } finally {
            target.destroy();
            invalid.destroy();
            await disposeGpuApps([app]);
            device.destroy();
        }
    });
}
