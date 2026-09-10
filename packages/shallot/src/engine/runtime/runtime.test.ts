import { describe, expect, test } from "bun:test";
import { now, Runtime, readBinary, readFile, requestFrame } from "./";

describe("Runtime", () => {
    test("detects Bun environment as headless", () => {
        expect(Runtime).toBe("headless");
    });

    test("reads files", async () => {
        const text = await readFile("./package.json");
        expect(text).toContain("shallot");

        const binary = await readBinary("./package.json");
        expect(binary).toBeInstanceOf(ArrayBuffer);
        expect(binary.byteLength).toBeGreaterThan(0);
    });

    test("provides timing", () => {
        const t1 = now();
        const t2 = now();
        expect(t2).toBeGreaterThanOrEqual(t1);
    });

    test("schedules frame callbacks", async () => {
        // resolve from inside the callback: the test waits on the frame actually
        // running, so an unscheduled callback fails by timeout, not by luck of timing
        let called = false;
        await new Promise<void>((resolve) =>
            requestFrame(() => {
                called = true;
                resolve();
            }),
        );
        expect(called).toBe(true);
    });
});
