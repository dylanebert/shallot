import { expect, spyOn, test } from "bun:test";
import { clear } from "../ecs/core";
import { observeDevice } from "../runtime/gpu";
import { type App, run } from ".";

function owner() {
    const loss = Promise.withResolvers<GPUDeviceLostInfo>();
    let fences = 0;
    const device = {
        queue: {
            onSubmittedWorkDone: async () => {
                fences++;
            },
        },
        features: new Set(),
        limits: {},
        lost: loss.promise,
        pushErrorScope: () => {},
        popErrorScope: async () => null,
    } as unknown as GPUDevice;
    const diagnostics: string[] = [];
    observeDevice(device, (message) => diagnostics.push(message));
    return { device, loss, diagnostics, fences: () => fences };
}

for (const mode of ["loss", "replacement", "same-device-rebuild"] as const) {
    test(`${mode} stops the original managed loop without stopping the replacement`, async () => {
        clear();
        const callbacks: (() => void)[] = [];
        const timer = spyOn(globalThis, "setTimeout").mockImplementation(((
            callback: () => void,
        ) => {
            callbacks.push(callback);
            return 0;
        }) as typeof setTimeout);
        const apps: App[] = [];
        const old = owner();
        const next = mode === "same-device-rebuild" ? old : owner();
        let oldSteps = 0;
        let nextSteps = 0;
        const loading = { show() {}, update() {} };
        try {
            apps.push(
                await run({
                    defaults: false,
                    device: old.device,
                    loading,
                    plugins: [{ name: "old", systems: [{ update: () => oldSteps++ }] }],
                }),
            );
            expect(callbacks).toHaveLength(1);
            callbacks.shift()!();
            expect(oldSteps).toBe(1);
            expect(old.fences()).toBe(1);
            apps.push(
                await run({
                    defaults: false,
                    device: next.device,
                    loading,
                    plugins: [{ name: "next", systems: [{ update: () => nextSteps++ }] }],
                }),
            );
            if (mode === "loss") {
                old.loss.resolve({ reason: "unknown", message: "old owner" } as GPUDeviceLostInfo);
                await old.loss.promise;
            }
            expect(callbacks).toHaveLength(2);
            callbacks.shift()!();
            expect(oldSteps).toBe(1);
            expect(next.fences()).toBe(next === old ? 1 : 0);
            expect(callbacks).toHaveLength(1);
            callbacks.shift()!();
            expect(nextSteps).toBe(1);
            expect(next.fences()).toBe(next === old ? 2 : 1);
            next.loss.resolve({ reason: "destroyed", message: "next owner" } as GPUDeviceLostInfo);
            await next.loss.promise;
            callbacks.shift()!();
            expect(nextSteps).toBe(1);
            expect(callbacks).toHaveLength(0);
            expect(old.diagnostics).toHaveLength(mode === "replacement" ? 0 : 1);
            expect(next.diagnostics).toHaveLength(1);
        } finally {
            for (const app of apps.reverse()) app.dispose();
            timer.mockRestore();
            clear();
        }
    });
}
