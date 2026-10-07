import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";
import { PhysicsWorld } from "./api/world";
import { clockImport, type Kernel, kernelState } from "./kernel/kernel";
import { mutationAllocationSubject } from "./mutation-allocation.fixture";

setDefaultTimeout(CEILING.node);
test("warmed type, filter, joint, event-flag and destroy/create mutations match native WASM allocation counts", async () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-mutation-alloc-"));
    const runtime = kernelState(undefined);
    const previous = runtime.instance;
    let world: PhysicsWorld | undefined;
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
        const bytes = await Bun.file(
            join(dir, "wasm32-unknown-unknown/release/shallot_physics.wasm"),
        ).arrayBuffer();
        const built = await WebAssembly.instantiate(bytes, {
            env: {
                now: clockImport(() => k.memory),
                queryCallback() {
                    throw new Error("unexpected mutation allocation callback");
                },
                materialCallback() {
                    throw new Error("unexpected mutation allocation material callback");
                },
            },
        });
        const k = built.instance.exports as unknown as Kernel & {
            allocationCount(): number;
            allocationControl(): void;
        };
        runtime.instance = k;
        world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
        const mutate = mutationAllocationSubject(world);
        for (let i = 0; i < 1200; ++i) mutate();
        const before = k.allocationCount();
        for (let i = 0; i < 600; ++i) mutate();
        // mutations.c at Box3D 47d7f7cc: island body/joint arrays allocate 1200 times for this sequence.
        expect(k.allocationCount() - before).toBe(1200);
        k.allocationControl();
        expect(k.allocationCount() - before).toBeGreaterThan(1200);
    } finally {
        world?.destroy();
        runtime.instance = previous;
        rmSync(dir, { recursive: true, force: true });
    }
});
test("warmed type, filter, joint, event-flag and destroy/create mutations allocate no JavaScript heap", async () => {
    const sample = await sampleAllocation(
        resolve(import.meta.dir, "fixtures/mutation-allocation.entry.ts"),
        { warm: 1200, frames: 600 },
    );
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});
