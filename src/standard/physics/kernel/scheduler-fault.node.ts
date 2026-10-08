import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../../scripts/test-tiers";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init, kernel, kernelState, shutdown } from "./kernel";
import { KERNEL_SHARED_WASM_BASE64, SHARED_STACK_SIZE } from "./kernel.shared.wasm";
import { createPool } from "./pool";

setDefaultTimeout(CEILING.node);

if (process.env.PHYSICS_SCHEDULER_FAULT === "1") {
    test("a wasm worker trap throws from the step and poisons further steps", async () => {
        await init(undefined, { threads: 4 });
        await shutdown(undefined);
        const k = kernel(undefined);
        const state = kernelState(undefined);
        const module = await WebAssembly.compile(Buffer.from(KERNEL_SHARED_WASM_BASE64, "base64"));
        const stack = (k as unknown as Record<string, WebAssembly.Global>).__stack_pointer;
        state.pool = await createPool(
            module,
            k.memory,
            3,
            stack.value as number,
            SHARED_STACK_SIZE,
            k,
            0,
        );
        state.resolved = 4;
        const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        for (let i = 0; i < 512; ++i) {
            world
                .createBody({
                    type: BodyType.Dynamic,
                    position: { x: i % 16, y: Math.floor(i / 16), z: 0 },
                })
                .createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.6 });
        }
        expect(() => world.step(1 / 60, 1)).toThrow("a physics worker faulted");
        expect(state.dead).toBe(true);
        expect(() => world.step(1 / 60, 1)).toThrow();
        await state.pool.terminate();
    });
} else {
    test("worker traps release the stepped kernel in an isolated process", async () => {
        const child = Bun.spawn([process.execPath, "test", import.meta.filename], {
            env: { ...process.env, PHYSICS_SCHEDULER_FAULT: "1" },
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(await child.exited).toBe(0);
    });
}
