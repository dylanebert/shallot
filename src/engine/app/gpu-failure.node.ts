import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { globals } from "webgpu";
import type { WorldGpu } from "../runtime";
import { runApp } from "./index";

setDefaultTimeout(TIMEOUT.node);

const { GPUValidationError, GPUOutOfMemoryError, GPUInternalError } = globals as Pick<
    typeof globalThis,
    "GPUValidationError" | "GPUOutOfMemoryError" | "GPUInternalError"
>;

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((yes) => (resolve = yes));
    return { promise, resolve };
}

for (const kind of ["Validation", "OutOfMemory", "Internal", "DeviceLost", "Firefox"] as const) {
    test(`${kind} stops runApp and raises one classified GPU error`, async () => {
        const lost = deferred<GPUDeviceLostInfo>();
        const events = new EventTarget();
        const device = {
            lost: lost.promise,
            pushErrorScope() {},
            popErrorScope: async () => null,
            addEventListener: events.addEventListener.bind(events),
        } as unknown as GPUDevice;
        const errors = spyOn(console, "error").mockImplementation(() => {});
        const warnings = spyOn(console, "warn").mockImplementation(() => {});
        const shown: unknown[] = [];
        const app = await runApp({
            defaults: false,
            plugins: [
                {
                    name: "mock GPU",
                    initialize(world) {
                        world.attachGpu({
                            adapter: {
                                class: "unidentified",
                                identity: "failure mock",
                                reason: "test",
                            },
                            device,
                            sync: async () => {},
                            pending: () => 0,
                            root: {} as WorldGpu["root"],
                            buffers: new Map(),
                            textures: new Map(),
                            samplers: new Map(),
                            typed: new Map(),
                            fences: { issued: 0, completed: 0 },
                            frame: 0,
                        } as WorldGpu);
                    },
                },
            ],
            loading: { show() {}, update() {}, error: (error) => shown.push(error) },
        });
        const step = spyOn(app.world, "step");
        const inject = (name: string) => {
            const event = new Event("uncapturederror");
            Object.defineProperty(event, "error", {
                value:
                    name === "GPUOutOfMemoryError"
                        ? new GPUOutOfMemoryError("Out of memory")
                        : name === "GPUInternalError"
                          ? new GPUInternalError("Internal error")
                          : new GPUValidationError("Validation error"),
            });
            events.dispatchEvent(event);
        };
        try {
            await Bun.sleep(30);
            expect(step.mock.calls.length).toBeGreaterThan(0);
            if (kind === "DeviceLost")
                lost.resolve({ reason: "unknown", message: "driver hung" } as GPUDeviceLostInfo);
            else inject(`GPU${kind === "Firefox" ? "Validation" : kind}Error`);
            const replay =
                kind === "Firefox"
                    ? Promise.all([
                          new Promise<void>((resolve) =>
                              setTimeout(() => {
                                  inject("GPUOutOfMemoryError");
                                  resolve();
                              }, 2),
                          ),
                          new Promise<void>((resolve) =>
                              setTimeout(() => {
                                  lost.resolve({
                                      reason: "unknown",
                                      message: "Out of memory",
                                  } as GPUDeviceLostInfo);
                                  resolve();
                              }, 143),
                          ),
                      ])
                    : Promise.resolve();
            await Promise.resolve();
            const frames = step.mock.calls.length;
            await Bun.sleep(30);
            expect(step.mock.calls.length).toBe(frames);
            await replay;
            await Bun.sleep(260);
            expect(errors).toHaveBeenCalledTimes(1);
            const error = errors.mock.calls[0]![0] as Error;
            expect(error).toBeInstanceOf(Error);
            expect(error.constructor.name).toBe("GpuError");
            expect(error.name).toBe(`Gpu${kind === "Firefox" ? "OutOfMemory" : kind}Error`);
            expect(shown).toEqual([error]);
            inject("GPUValidationError");
            await Bun.sleep(10);
            expect(errors).toHaveBeenCalledTimes(1);
            expect(warnings).toHaveBeenCalledTimes(1);
        } finally {
            app.dispose();
            errors.mockRestore();
            warnings.mockRestore();
        }
    });
}
