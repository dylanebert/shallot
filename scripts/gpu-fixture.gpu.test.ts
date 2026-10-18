import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { rawDevice } from "../src/engine/runtime";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "./gpu.fixture";
import { expectError } from "./test-console";

setDefaultTimeout(TIMEOUT.gpu);
const subject = compileGpuFile(import.meta.path, async () => {
    const owner = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "FixtureOwner", gpu: {} }],
    });
    const device = rawDevice(owner.world.gpu.device);
    const first = await createTestGpuApp({ defaults: false, plugins: [], device });
    const second = await createTestGpuApp({ defaults: false, plugins: [], device });
    return { owner, device, first, second };
});

test("fixture release fences consumers, preserves borrowed devices and remembers disposed owners", async () => {
    const { owner, device, first, second } = subject();
    const destroy = device.destroy.bind(device);
    let destroys = 0;
    device.destroy = () => {
        destroys++;
        destroy();
    };
    await disposeGpuApps([first]);
    expect(destroys).toBe(0);
    const buffer = second.world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, new Uint32Array([17]));
    await disposeGpuApps([second]);
    expect(destroys).toBe(0);
    const supplied = new Proxy(device, {
        get(target, key) {
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    const external = await createTestGpuApp({ defaults: false, plugins: [], device: supplied });
    await disposeGpuApps([external]);
    expect(destroys).toBe(0);

    const consumer = await createTestGpuApp({ defaults: false, plugins: [], device });
    owner.world.dispose();
    owner.dispose();
    const fence = device.queue.onSubmittedWorkDone.bind(device.queue);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
        release = resolve;
    });
    device.queue.onSubmittedWorkDone = async () => {
        await fence();
        await gate;
    };
    const events: string[] = [];
    const ownerDispose = owner.dispose.bind(owner);
    owner.dispose = () => {
        events.push("owner");
        ownerDispose();
    };
    const consumerDispose = consumer.dispose.bind(consumer);
    consumer.dispose = () => {
        events.push("consumer");
        consumerDispose();
    };
    device.destroy = () => {
        expect(events).toEqual(["consumer", "owner"]);
        destroys++;
        destroy();
    };
    const apps = [owner, consumer];
    // The supplied proxy has its own device-loss observer.
    expectError(/^GPU device lost destroyed: device was destroyed$/);
    const pending = disposeGpuApps(apps);
    await Promise.resolve();
    expect(destroys).toBe(0);
    expect(events).toEqual([]);
    release();
    await pending;
    expect(destroys).toBe(1);
    expect(apps[0]).toBe(owner);
    expect(apps[1]).toBe(consumer);
    await disposeGpuApps([owner, consumer]);
    expect(destroys).toBe(1);
});
