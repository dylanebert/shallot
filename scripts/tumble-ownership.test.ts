import { expect, test } from "bun:test";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");

// Separate processes prove both initialization modes without inheriting another test's kernel.
for (const threads of [0, 2]) {
    test(`linked adapter and public core share the canonical solver (${threads})`, () => {
        const child = Bun.spawnSync(
            [
                "bun",
                "--preload",
                "./packages/shallot/tests/setup.ts",
                "-e",
                `
import assert from "node:assert/strict";
import { State, Tumble, TumblePlugin } from "@dylanebert/shallot";
import { World, init, threads } from "@dylanebert/shallot/tumble/core";
import { World as CanonicalWorld } from "./packages/shallot-tumble/src/standard/tumble/engine/index.ts";
import { kernel, workers } from "./packages/shallot-tumble/src/standard/tumble/engine/kernel.ts";
assert.strictEqual(World, CanonicalWorld, "canonical solver identity");
await init({ threads: ${threads} });
const before = { kernel: kernel(), pool: workers() };
const state = new State();
await TumblePlugin.warm(state);
assert(Tumble.world instanceof CanonicalWorld, "adapter solver identity");
assert.strictEqual(kernel(), before.kernel, "canonical kernel identity");
assert.strictEqual(workers(), before.pool, "canonical pool identity");
assert.equal(threads(), ${threads || 1});
TumblePlugin.dispose(state);
state.dispose();
console.log("SOLVER_OWNER_IDENTITY_OK");
`,
            ],
            { cwd: root, stdout: "pipe", stderr: "pipe", timeout: 15000 },
        );
        console.log(
            `linked solver threads=${threads} exit=${child.exitCode}\n${child.stdout}${child.stderr}`,
        );
        expect(child.signalCode).toBeUndefined();
        expect(child.stderr.toString()).not.toContain("AssertionError");
        expect(child.exitCode).toBe(0);
        expect(child.stdout.toString()).toContain("SOLVER_OWNER_IDENTITY_OK");
    });
}
