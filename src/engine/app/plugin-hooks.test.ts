import { expect, test } from "bun:test";
import type { Plugin } from "./index";
import { createApp } from "./index";

function captureUserTiming(): {
    marks: string[];
    measures: string[];
    measureStarts: string[];
    setMarkThrowing(throwing: boolean): void;
    restore(): void;
} {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "performance");
    const original = globalThis.performance;
    const marks: string[] = [];
    const measures: string[] = [];
    const measureStarts: string[] = [];
    let markThrows = false;
    const timing = Object.create(original) as Performance;
    Object.defineProperties(timing, {
        mark: {
            configurable: true,
            value(name: string) {
                if (markThrows) throw new Error("performance.mark unavailable");
                marks.push(name);
            },
        },
        measure: {
            configurable: true,
            value(name: string, start?: string) {
                measures.push(name);
                if (start) measureStarts.push(start);
            },
        },
    });
    Object.defineProperty(globalThis, "performance", {
        configurable: true,
        writable: true,
        value: timing,
    });
    return {
        marks,
        measures,
        measureStarts,
        setMarkThrowing(throwing) {
            markThrows = throwing;
        },
        restore() {
            if (descriptor) Object.defineProperty(globalThis, "performance", descriptor);
            else Reflect.deleteProperty(globalThis, "performance");
        },
    };
}

test("plugin initialize and warm hooks write guarded User Timing spans", async () => {
    const timing = captureUserTiming();
    try {
        const first: Plugin = {
            name: "TraceFirst",
            initialize() {},
            warm() {},
        };
        const second: Plugin = {
            name: "TraceSecond",
            dependencies: [first],
            async initialize() {},
            warm() {},
        };
        const app = await createApp({ defaults: false, plugins: [second, first] });
        app.dispose();

        const firstInitialize = "shallot:plugin:TraceFirst:initialize";
        const secondInitialize = "shallot:plugin:TraceSecond:initialize";
        expect(timing.marks.filter((name) => name.endsWith(":initialize"))).toEqual([
            firstInitialize,
            secondInitialize,
        ]);
        expect(timing.measures.filter((name) => name.endsWith(":initialize"))).toEqual([
            firstInitialize,
            secondInitialize,
        ]);
        expect(timing.measureStarts.filter((name) => name.endsWith(":initialize"))).toEqual([
            firstInitialize,
            secondInitialize,
        ]);
        for (const name of ["shallot:plugin:TraceFirst:warm", "shallot:plugin:TraceSecond:warm"]) {
            expect(timing.marks).toContain(name);
            expect(timing.measures).toContain(name);
            expect(timing.measureStarts).toContain(name);
        }

        const throwingName = "shallot:plugin:TraceThrow:initialize";
        let failure: unknown;
        try {
            await createApp({
                defaults: false,
                plugins: [
                    {
                        name: "TraceThrow",
                        initialize() {
                            throw new Error("expected failure");
                        },
                    },
                ],
            });
        } catch (error) {
            failure = error;
        }
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toContain('Plugin "TraceThrow" initialize threw');
        expect(timing.marks).toContain(throwingName);
        expect(timing.measures).toContain(throwingName);
        expect(timing.measureStarts).toContain(throwingName);

        timing.setMarkThrowing(true);
        const resilient = await createApp({
            defaults: false,
            plugins: [{ name: "TraceUnavailable", initialize() {}, warm() {} }],
        });
        resilient.dispose();
    } finally {
        timing.restore();
    }
});

test("initialize progress reaches Loading.update before an awaited hook resolves", async () => {
    const updates: number[] = [];
    let updatedBeforeResolve = false;
    let hookResolved = false;
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "ProgressSlices",
                async initialize(_world, onProgress) {
                    await new Promise<void>((resolve) => setTimeout(resolve, 0));
                    onProgress?.(0.5);
                    updatedBeforeResolve = updates.includes(0.5) && !hookResolved;
                    hookResolved = true;
                },
            },
        ],
        loading: {
            show() {},
            update(progress) {
                updates.push(progress);
            },
        },
    });
    app.dispose();

    expect(updatedBeforeResolve).toBe(true);
    expect(updates).toContain(0.5);
});
