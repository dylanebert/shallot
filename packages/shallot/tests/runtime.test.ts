import { test, expect, describe } from "bun:test";
import { Runtime, now, requestFrame, readFile, readBinary } from "../src/engine/runtime";

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
        let called = false;
        requestFrame(() => {
            called = true;
        });
        await new Promise((resolve) => setTimeout(resolve, 10));
        expect(called).toBe(true);
    });
});
