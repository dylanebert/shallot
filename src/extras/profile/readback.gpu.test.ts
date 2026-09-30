import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Profile, ProfilePlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [ProfilePlugin] }]);

test("a profiler without timestamp-query runs and distinguishes missing GPU timings from zero", async () => {
    const app = subjects()[0];
    try {
        expect(app.state.gpu.device.features.has("timestamp-query")).toBe(false);
        const stats = app.state.resource(Profile);
        app.state.step(0);
        expect(stats.gpuTiming).toBe("requires timestamp-query");
        expect(app.state.gpu.span?.("untimed")).toBeUndefined();
        expect(stats.gpu.size).toBe(0);
        expect(stats.gpuTime.size).toBe(0);
        expect(stats.gpuFires.size).toBe(0);
        expect(app.state.readback.allocated).toBe(0);
        const before = stats.bufferBytes;
        const buffer = app.state.gpu.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
        });
        expect(stats.bufferBytes).toBe(before + 16);
        buffer.destroy();
        expect(stats.bufferBytes).toBe(before);
    } finally {
        app.dispose();
    }
});
