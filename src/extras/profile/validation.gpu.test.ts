import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { DiagnosticsStore } from "../../engine";
import { ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const subject = compileGpuFile(import.meta.path, async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("profiler adapter unavailable");
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
    console.info(JSON.stringify({ adapter: adapter.info, features: [...device.features] }));
    return { adapter, device };
});

test("rejected timestamp validation leaves every frame valid and GPU timing unavailable", async () => {
    const { adapter, device } = subject();
    const createEncoder = device.createCommandEncoder.bind(device);
    const pop = device.popErrorScope.bind(device);
    const invalid = device.createQuerySet({ type: "occlusion", count: 2 });
    let probe = false;
    device.createCommandEncoder = (descriptor) => {
        if (descriptor?.label === "profile-probe") probe = true;
        const encoder = createEncoder(descriptor);
        const begin = encoder.beginComputePass.bind(encoder);
        encoder.beginComputePass = (desc) =>
            begin(
                desc?.timestampWrites
                    ? { ...desc, timestampWrites: { ...desc.timestampWrites, querySet: invalid } }
                    : desc,
            );
        return encoder;
    };
    device.popErrorScope = async () => {
        if (!probe) return pop();
        probe = false;
        await pop();
        return new GPUValidationError("timestamp writes rejected");
    };
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [ProfilePlugin],
        device,
        adapter,
    });
    device.popErrorScope = pop;
    try {
        for (let frame = 0; frame < 6; frame++) {
            expect(app.world.gpu.span!("unavailable")).toBeUndefined();
            device.pushErrorScope("validation");
            app.world.step(1 / 60);
            const encoder = device.createCommandEncoder();
            encoder.beginComputePass({ timestampWrites: app.world.gpu.span!("frame") }).end();
            device.queue.submit([encoder.finish()]);
            expect(await pop()).toBeNull();
            expect(
                app.world.resource(DiagnosticsStore).get("render/timestamp_query")?.value(),
            ).toBe(0);
        }
    } finally {
        await disposeGpuApps([app]);
        device.createCommandEncoder = createEncoder;
        invalid.destroy();
        device.destroy();
    }
});
