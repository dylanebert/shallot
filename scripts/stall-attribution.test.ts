import { expect, test } from "bun:test";
import { assertCapture, main } from "./stall-attribution";
import type { VerifyResult } from "./verify";

const clean = {
    pass: true,
    attribution: { userAgent: "Mozilla/5.0 Chrome/152.0" },
    cpuProfile: {
        totalMs: 1,
        entries: [{ key: "frame", functionName: "app", url: "", lineNumber: 0, selfMs: 1 }],
        buckets: [
            {
                name: "app — in-page evaluated script, no source url (not a native frame: page.evaluate serializes a function with no file, so CDP reports an empty url under the function's own name)",
                selfMs: 1,
            },
        ],
    },
} as VerifyResult;
const contaminated = {
    ...clean,
    cpuProfile: {
        totalMs: 1,
        entries: [
            { key: "decode", functionName: "decodeSample", url: "", lineNumber: 0, selfMs: 1 },
        ],
        buckets: [
            {
                name: "decodeSample — in-page evaluated script, no source url (not a native frame: page.evaluate serializes a function with no file, so CDP reports an empty url under the function's own name)",
                selfMs: 1,
            },
        ],
    },
};

test("captured result guards discriminate missing, empty, invalid, headless and contaminated", () => {
    expect(() => assertCapture(clean)).not.toThrow();
    for (const result of [
        null,
        { pass: false },
        { ...clean, attribution: undefined },
        { ...clean, attribution: { userAgent: " " } },
        { ...clean, attribution: { userAgent: "HeadlessChrome/152" } },
        { ...clean, cpuProfile: undefined },
        { ...clean, cpuProfile: null },
        ...[0, -1, NaN, Infinity].map((totalMs) => ({
            ...clean,
            cpuProfile: { ...clean.cpuProfile!, totalMs },
        })),
        { ...clean, cpuProfile: { ...clean.cpuProfile!, entries: [] } },
        { ...clean, cpuProfile: { ...clean.cpuProfile!, buckets: [] } },
        { ...clean, cpuProfile: { ...clean.cpuProfile!, entries: [null] } },
        {
            ...contaminated,
            cpuProfile: { ...contaminated.cpuProfile, buckets: [{ name: "unrelated", selfMs: 1 }] },
        },
        contaminated,
    ])
        expect(() => assertCapture(result as VerifyResult | null)).toThrow();
    expect(() => assertCapture(contaminated)).toThrow("harness contamination");
});

test("driver binds the wrapper arguments and refuses before capture without display", async () => {
    const calls: unknown[] = [];
    const deps = {
        skipReason: () => null,
        verify: async (dir: string, flags?: string[]) => {
            calls.push([dir, flags]);
            return clean;
        },
    };
    expect(await main([], deps)).toBe(0);
    expect(await main(["--dir", "fixture", "--query", "x=1", "--query", "y=2"], deps)).toBe(0);
    expect(calls).toEqual([
        ["examples/showcase/sandbox", ["--attribution", "--timeout", "30000"]],
        ["fixture", ["--attribution", "--timeout", "30000", "--query", "x=1", "--query", "y=2"]],
    ]);
    expect(await main([], { ...deps, skipReason: () => "no display" })).toBe(1);
    expect(calls.length).toBe(2);
    expect(await main(["--dir"], deps)).toBe(1);
    expect(
        await main([], {
            ...deps,
            verify: async () => {
                throw new Error("capture failed");
            },
        }),
    ).toBe(1);
});

test("driver result is consumed as a real process exit, not a live capture receipt", async () => {
    for (const [result, exit, message] of [
        [clean, 0, "No registered harness frames"],
        [contaminated, 1, "harness contamination"],
        [{ ...clean, attribution: { userAgent: "HeadlessChrome/152" } }, 1, "HeadlessChrome"],
        [null, 1, "verify failed"],
    ] as const) {
        const proc = Bun.spawn(
            [
                "bun",
                "-e",
                `import {main} from ${JSON.stringify(import.meta.dir + "/stall-attribution.ts")}; process.exitCode = await main([], {skipReason:()=>null, verify:async()=>(${JSON.stringify(result)})});`,
            ],
            { stdout: "pipe", stderr: "pipe" },
        );
        const [stdout, stderr, code] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        expect({ code, output: stdout + stderr }).toEqual({
            code: exit,
            output: expect.stringContaining(message),
        });
        expect(stdout + stderr).toContain(message);
    }
});
