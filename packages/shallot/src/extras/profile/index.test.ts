import { describe, expect, test } from "bun:test";
import { Compute, State } from "../../engine";
import { UnsupportedError } from "../../engine/runtime";
import { ProfilePlugin } from "./index";

// `ProfilePlugin.features` declares `timestamp-query`, and `acquireDevice` throws on a device that can't
// grant it — but `requestGPU(externalDevice)` / `run({ device })` adopts a device as-is, so a declared
// feature never reaches an adopted one. `attach` is the layer both paths cross: it owns the engine's only
// `createQuerySet`, and without a guard there it would issue one against a device that can't validate it,
// leaving a raw GPU error on `onuncapturederror` and a profiler reporting silent zeros.
const device = (...features: string[]) =>
    ({ features: { has: (f: string) => features.includes(f) } }) as unknown as GPUDevice;

describe("ProfilePlugin", () => {
    test("an adopted device without timestamp-query fails loud, naming it", () => {
        const prev = Compute.device;
        Object.assign(Compute, { device: device("indirect-first-instance") });
        let caught: unknown;
        try {
            ProfilePlugin.initialize?.(new State());
        } catch (e) {
            caught = e;
        } finally {
            Object.assign(Compute, { device: prev });
        }
        expect(caught).toBeInstanceOf(UnsupportedError);
        expect((caught as UnsupportedError).missing).toEqual(["timestamp-query"]);
    });

    test("a re-attach to a device missing timestamp-query still fails loud", () => {
        const prev = Compute.device;
        const capable = {
            features: { has: () => true },
            createQuerySet: () => ({}),
            createBuffer: () => ({ destroy() {} }),
            createTexture: () => ({ destroy() {} }),
            createComputePipelineAsync: async () => ({}),
            createRenderPipelineAsync: async () => ({}),
            queue: { submit() {} },
        } as unknown as GPUDevice;
        let caught: unknown;
        try {
            Object.assign(Compute, { device: capable });
            ProfilePlugin.initialize?.(new State());
            Object.assign(Compute, { device: device("indirect-first-instance") });
            ProfilePlugin.initialize?.(new State());
        } catch (e) {
            caught = e;
        } finally {
            Object.assign(Compute, { device: prev });
        }
        expect(caught).toBeInstanceOf(UnsupportedError);
        expect((caught as UnsupportedError).missing).toEqual(["timestamp-query"]);
    });
});
