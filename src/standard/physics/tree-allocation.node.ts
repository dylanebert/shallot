import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";

import { countedKernelBytes } from "../../../scripts/counted-kernel";
import { clockImport, type Kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);
let kernelBytes: ArrayBuffer;
beforeAll(async () => {
    kernelBytes = await countedKernelBytes();
}, TIMEOUT.startup);
const entry = resolve(import.meta.dir, "fixtures/tree-allocation.entry.ts");
test("tree queries allocate no WASM heap and destroying trees returns live bytes to baseline", async () => {
    const built = await WebAssembly.instantiate(kernelBytes, {
        env: {
            now: clockImport(() => k.memory),
            queryCallback: () => 1,
            materialCallback() {
                throw new Error("unexpected material callback");
            },
            collisionCallback() {
                throw new Error("unexpected collision callback");
            },
        },
    });
    const k = built.instance.exports as unknown as Kernel & {
        allocationCount(): number;
        allocationLiveBytes(): number;
        allocationControl(): void;
    };
    const baseline = k.allocationLiveBytes();
    const tree = k.standaloneTreeCreate(16);
    try {
        const words = new Float32Array(k.memory.buffer, k.standaloneTreeWords(tree), 32);
        words.set([-1, -1, -1, 1, 1, 1]);
        for (let i = 0; i < 16; i++) k.standaloneTreeCall(tree, 0, 0, 0, 1, 0, i, 0, 0);
        const query = () => {
            for (const op of [4, 5, 6])
                k.standaloneTreeCall(tree, op, 0, 0xffffffff, 0xffffffff, 0, 0, 0, 100);
        };
        query();
        const count = k.allocationCount(),
            bytes = k.allocationLiveBytes();
        for (let i = 0; i < 600; i++) query();
        expect(k.allocationCount()).toBe(count);
        expect(k.allocationLiveBytes()).toBe(bytes);
        k.allocationControl();
        expect(k.allocationCount()).toBeGreaterThan(count);
        // A deliberately retained tree must fail the live-byte baseline.
        expect(k.allocationLiveBytes()).toBeGreaterThan(baseline);
    } finally {
        k.standaloneTreeDestroy(tree);
    }
    expect(k.allocationLiveBytes()).toBe(baseline);
});
test("warmed independent tree overlap, ray and closest queries allocate nothing", async () => {
    const sample = await sampleAllocation(entry, { warm: 12000 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
test("independent tree allocation gate rejects copied query records", async () => {
    const sample = await sampleAllocation(entry, { warm: 12000, input: "leak" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("allocated JavaScript heap");
});
