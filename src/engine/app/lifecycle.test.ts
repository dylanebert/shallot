import { expect, test } from "bun:test";
import { Time } from "../ecs";
import type { Plugin } from "./index";
import { createApp, runApp } from "./index";

function withoutGpu(): () => void {
    const previous = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: undefined });
    return () => {
        if (previous) Object.defineProperty(navigator, "gpu", previous);
        else Reflect.deleteProperty(navigator, "gpu");
    };
}

test("a fixed-system composition builds and steps through both app lifecycles without WebGPU", async () => {
    const restore = withoutGpu();
    let ticks = 0;
    let warms = 0;
    const plugin: Plugin = {
        name: "CpuLifecycle",
        recovery: "stateless",
        systems: [{ name: "cpu-tick", group: "fixed", update: () => ticks++ }],
        warm() {
            warms++;
        },
    };
    try {
        const built = await createApp({ defaults: false, plugins: [plugin] });
        built.world.step(Time.FIXED_DT);
        expect(ticks).toBe(1);
        expect(warms).toBe(1);
        expect(built.world.frameFence).toBeUndefined();
        expect(() => built.world.gpu).toThrow("no enabled plugin declares a GPU requirement");
        built.dispose();

        ticks = 0;
        let running: Awaited<ReturnType<typeof runApp>> | undefined;
        try {
            const app = await runApp({ defaults: false, plugins: [plugin] });
            running = app;
            await new Promise<void>((resolve, reject) => {
                const poll = setInterval(() => {
                    if (ticks > 0) {
                        clearInterval(poll);
                        clearTimeout(timeout);
                        resolve();
                    }
                }, 0);
                const timeout = setTimeout(() => {
                    clearInterval(poll);
                    reject(new Error("runApp did not step its CPU-only world"));
                }, 100);
            });
            expect(ticks).toBeGreaterThan(0);
            expect(warms).toBe(2);
            expect(app.world.frameFence).toBeUndefined();
            expect(() => app.world.gpu).toThrow("no enabled plugin declares a GPU requirement");
        } finally {
            running?.dispose();
        }
    } finally {
        restore();
    }
});
