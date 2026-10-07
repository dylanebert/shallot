import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { CEILING } from "../../scripts/test-tiers";
import { allocatesNothing, allocationFailure, sampleAllocation, windowBytes } from "./allocation";

setDefaultTimeout(CEILING.node);

const PLANTED = resolve(import.meta.dir, "planted.entry.ts");

test("a first optimization inside a window refuses its byte reading and names the function", async () => {
    const sample = await sampleAllocation(PLANTED, { warm: 120, frames: 120, input: "compile" });
    expect(sample.windows[0].optimizations).toContain("lateCompile");
    expect(sample.windows[0].sites).toEqual([]);
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toContain("after warm 120; optimized");
    expect(allocationFailure(sample)).toContain("lateCompile");
    expect(allocationFailure(sample)).toContain("no byte reading");
});

test("a planted steady literal is reported as bytes", async () => {
    const sample = await sampleAllocation(PLANTED, { warm: 120, frames: 120, input: "steady" });
    expect(sample.windows.every((window) => !window.optimizations?.length && windowBytes(window) > 0)).toBe(true);
    expect(allocationFailure(sample)).toContain("steady play allocated JavaScript heap");
});
