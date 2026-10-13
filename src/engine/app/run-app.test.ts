import { expect, test } from "bun:test";
import { runApp } from "./index";

test("runApp steps a callback under half the median cadence but holds six pending frames", async () => {
    const callbacks: (() => void)[] = [];
    const originalSetTimeout = globalThis.setTimeout;
    const originalNow = Object.getOwnPropertyDescriptor(performance, "now");
    let time = 0;
    let pending = 0;
    let draws = 0;
    Object.defineProperty(performance, "now", { configurable: true, value: () => time });
    globalThis.setTimeout = ((callback: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (typeof callback === "function" && delay === 0) {
            callbacks.push(() => callback(...args));
            return 0 as unknown as ReturnType<typeof setTimeout>;
        }
        return originalSetTimeout(callback, delay, ...args);
    }) as typeof setTimeout;

    let app: Awaited<ReturnType<typeof runApp>> | undefined;
    try {
        app = await runApp({
            defaults: false,
            plugins: [
                {
                    name: "FrameScheduleProbe",
                    systems: [{ group: "draw", update: () => draws++ }],
                },
            ],
            setup(world) {
                Object.defineProperty(world, "gpuIfAvailable", {
                    configurable: true,
                    value: { pending: () => pending },
                });
            },
        });

        const callback = (timestamp: number, framesPending = 0) => {
            const next = callbacks.shift();
            if (!next) throw new Error("runApp did not schedule the next frame");
            time = timestamp;
            pending = framesPending;
            next();
        };

        callback(0);
        callback(16);
        expect(draws).toBe(2);

        callback(17);
        expect(draws).toBe(3);

        callback(40, 6);
        expect(draws).toBe(3);
    } finally {
        app?.dispose();
        globalThis.setTimeout = originalSetTimeout;
        if (originalNow) Object.defineProperty(performance, "now", originalNow);
        else Reflect.deleteProperty(performance, "now");
    }
});
