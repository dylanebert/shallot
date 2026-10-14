import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { bunChild } from "../../../../scripts/bun-child";
import { BodyType } from "../common/types";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, stepTestWorld, testShapeDef } from "../testing";
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
        const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        for (let i = 0; i < 512; ++i) {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i % 16, y: Math.floor(i / 16), z: 0 },
            });
            createSphereShape(world, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.6,
            });
        }
        expect(() => stepTestWorld(world, 1 / 60, 1)).toThrow("a physics worker faulted");
        expect(state.dead).toBe(true);
        expect(() => stepTestWorld(world, 1 / 60, 1)).toThrow();
        await state.pool.terminate();
    });
} else {
    test("worker traps release the stepped kernel in an isolated process", async () => {
        const child = Bun.spawn({
            ...bunChild(["test", import.meta.filename]),
            env: { ...process.env, PHYSICS_SCHEDULER_FAULT: "1" },
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(await child.exited).toBe(0);
    });
}
