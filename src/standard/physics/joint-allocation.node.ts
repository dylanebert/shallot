import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { removeTestDirectory } from "../../../scripts/test-directory";
import { jointAllocationSubject, splitAllocationSubject } from "./joint-allocation.fixture";
import { clockImport, type Kernel, kernelState } from "./kernel/kernel";
import { createTestWorld, destroyTestWorld } from "./testing";
import type { WorldState } from "./world/world";

setDefaultTimeout(TIMEOUT.node);
const entry = resolve(import.meta.dir, "fixtures/joint-allocation.entry.ts");
let countedKernelBytes: ArrayBuffer | undefined;

beforeAll(async () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-joint-alloc-"));
    try {
        const build = Bun.spawnSync(
            [
                "cargo",
                "build",
                "-p",
                "shallot-physics",
                "--release",
                "--target",
                "wasm32-unknown-unknown",
                "--features",
                "count-allocations",
                "--target-dir",
                dir,
            ],
            { cwd: resolve(import.meta.dir, "../../..") },
        );
        if (build.exitCode !== 0) throw new Error(build.stderr.toString());
        countedKernelBytes = await Bun.file(
            join(dir, "wasm32-unknown-unknown/release/shallot_physics.wasm"),
        ).arrayBuffer();
    } finally {
        removeTestDirectory(dir);
    }
}, TIMEOUT.startup);
const fieldEntry = resolve(import.meta.dir, "fixtures/joint-field-allocation.entry.ts");

test("a public ECS motor-speed write allocates no steady JavaScript heap", async () => {
    const sample = await sampleAllocation(fieldEntry, { warm: 6000, frames: 600 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});

test("the public ECS motor-speed allocation check detects an allocating control", async () => {
    const sample = await sampleAllocation(fieldEntry, {
        warm: 1200,
        frames: 600,
        input: "allocating",
    });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("steady play allocated JavaScript heap");
});

test("the same warm joint subject allocates no WASM heap and its counting allocator detects an allocating control", async () => {
    const bytes = countedKernelBytes;
    if (!bytes) throw new Error("counting kernel was not built");
    const runtime = kernelState(undefined);
    const previous = runtime.instance;
    let world: WorldState | undefined;
    let splitWorld: WorldState | undefined;
    try {
        const built = await WebAssembly.instantiate(bytes, {
            env: {
                now: clockImport(() => k.memory),
                queryCallback() {
                    throw new Error("unexpected allocation-subject callback");
                },
                materialCallback() {
                    throw new Error("unexpected allocation-subject material callback");
                },
                collisionCallback() {
                    throw new Error("unexpected allocation-subject collision callback");
                },
            },
        });
        const k = built.instance.exports as unknown as Kernel & {
            allocationCount(): number;
            allocationControl(): void;
        };
        runtime.instance = k;
        world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
        const step = jointAllocationSubject(world);
        splitWorld = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        const split = splitAllocationSubject(splitWorld);
        for (let i = 0; i < 1200; ++i) {
            step();
            split();
        }
        const before = k.allocationCount();
        for (let i = 0; i < 600; ++i) {
            step();
            split();
        }
        expect(k.allocationCount() - before).toBe(0);
        k.allocationControl();
        expect(k.allocationCount() - before).toBeGreaterThan(0);
    } finally {
        if (splitWorld) destroyTestWorld(splitWorld);
        if (world) destroyTestWorld(world);
        runtime.instance = previous;
    }
});

test("kernel joint lifecycle, plain-id fields, internal events and pair finding with jointed bodies allocate no steady JavaScript heap", async () => {
    const sample = await sampleAllocation(entry, { warm: 6000, frames: 600 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});

test("the joint allocation assertion rejects an allocating lifecycle control", async () => {
    const sample = await sampleAllocation(entry, { warm: 1200, frames: 600, input: "allocating" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("steady play allocated JavaScript heap");
});
