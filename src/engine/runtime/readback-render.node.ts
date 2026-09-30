import { expect, setDefaultTimeout, test } from "bun:test";
import createRenderedSubject from "../../../diagnostics/readback-allocation/render.entry";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../app";
import { rawDevice } from "./gpu";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("rendered frames without a request map nothing", async () => {
    const owner = await createApp({ defaults: false, plugins: [] });
    const device = rawDevice(owner.world.gpu.device);
    const original = device.createBuffer.bind(device);
    let maps = 0;
    device.createBuffer = (descriptor) => {
        const buffer = original(descriptor);
        const map = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = (...args) => {
            maps++;
            return map(...args);
        };
        return buffer;
    };
    let subject: Awaited<ReturnType<typeof createRenderedSubject>> | undefined;
    try {
        subject = await createRenderedSubject("", device);
        for (let i = 0; i < 482; i++) subject.step();
        await subject.wait();
        expect(maps).toBe(0);
    } finally {
        device.createBuffer = original;
        subject?.dispose();
        owner.dispose();
    }
});
