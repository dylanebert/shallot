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

// A frame entered mid-step models both a nested event loop and Firefox's navigation cancel, which ends
// a frame without unwinding it; only the nested case returns to finish its step.
test("runApp skips a frame that starts while the world is mid-step", async () => {
    const callbacks: (() => void)[] = [];
    const originalSetTimeout = globalThis.setTimeout;
    const originalError = console.error;
    const errors: unknown[][] = [];
    let draws = 0;
    let reenter = false;
    console.error = (...args: unknown[]) => errors.push(args);
    globalThis.setTimeout = ((callback: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (typeof callback === "function" && delay === 0) {
            callbacks.push(() => callback(...args));
            return 0 as unknown as ReturnType<typeof setTimeout>;
        }
        return originalSetTimeout(callback, delay, ...args);
    }) as typeof setTimeout;
    const next = () => {
        const callback = callbacks.shift();
        if (!callback) throw new Error("runApp did not schedule the next frame");
        callback();
    };

    let app: Awaited<ReturnType<typeof runApp>> | undefined;
    try {
        app = await runApp({
            defaults: false,
            plugins: [
                {
                    name: "ReentryProbe",
                    systems: [
                        {
                            group: "draw",
                            update: () => {
                                draws++;
                                if (reenter) {
                                    reenter = false;
                                    next();
                                }
                            },
                        },
                    ],
                },
            ],
        });

        next();
        reenter = true;
        next();
        expect(draws).toBe(2);
        expect(errors).toEqual([]);

        next();
        next();
        expect(draws).toBe(4);
        expect(errors).toEqual([]);
    } finally {
        app?.dispose();
        console.error = originalError;
        globalThis.setTimeout = originalSetTimeout;
    }
});
