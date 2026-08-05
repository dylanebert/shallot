import { describe, expect, test } from "bun:test";
import { Compute, State } from "../../engine";
import { UnsupportedError } from "../../engine/runtime";
import { Profile, ProfilePlugin } from "./index";

// `ProfilePlugin.features` declares `timestamp-query`, and `acquireDevice` throws on a device that can't
// grant it — but `requestGPU(externalDevice)` / `run({ device })` adopts a device as-is, so a declared
// feature never reaches an adopted one. `attach` is the layer both paths cross: it owns the engine's only
// `createQuerySet`, and without a guard there it would issue one against a device that can't validate it,
// leaving a raw GPU error on `onuncapturederror` and a profiler reporting silent zeros.
const device = (...features: string[]) =>
    ({ features: { has: (f: string) => features.includes(f) } }) as unknown as GPUDevice;

// a device that grants everything `attach` touches — enough to install the plugin's Compute sinks
const capableDevice = () =>
    ({
        features: { has: () => true },
        createQuerySet: () => ({}),
        createBuffer: () => ({ destroy() {} }),
        createTexture: () => ({ destroy() {} }),
        createComputePipelineAsync: async () => ({}),
        createRenderPipelineAsync: async () => ({}),
        queue: { submit() {}, onSubmittedWorkDone: () => Promise.resolve() },
    }) as unknown as GPUDevice;

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

    // `attach` skips re-patching once `_querySet` exists (the reload-safety guard against re-creating GPU
    // resources on a live singleton), so it only ever installs the `createBuffer`/`createTexture` byte
    // tracking on the FIRST device this module-lifetime singleton attaches to. This test earns that slot by
    // running first among the successful attaches in this file — see the comment on the test below.
    //
    // The overlay's `collectStats` reads `profile.bufferBytes` / `profile.textureBytes` / `profile.sizes`
    // directly (index.ts `collectStats`) — the byte-budget gate's `Profile.bufferBytes` / `Profile.allocBytes`
    // must be that same field, not a parallel derivation the two could drift apart on. Pinning the public
    // seam against a live allocate + destroy is the differential.
    test("the public byte totals are the overlay's own fields, and destroy() decrements them", () => {
        const prev = Compute.device;
        Object.assign(Compute, { device: capableDevice() });
        try {
            ProfilePlugin.initialize?.(new State());
            const buffer = Compute.device.createBuffer({
                label: "shallot-perf-gates-test",
                size: 1024,
                usage: GPUBufferUsage.STORAGE,
            });

            expect(Profile.bufferBytes).toBeGreaterThanOrEqual(1024);
            expect(Profile.allocBytes.get("shallot-perf-gates-test")).toBe(1024);

            const before = Profile.bufferBytes;
            buffer.destroy();
            expect(Profile.bufferBytes).toBe(before - 1024);
            expect(Profile.allocBytes.has("shallot-perf-gates-test")).toBe(false);
        } finally {
            Object.assign(Compute, { device: prev });
        }
    });

    test("a re-attach to a device missing timestamp-query still fails loud", () => {
        const prev = Compute.device;
        const capable = capableDevice();
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

    // typegpu pipelines are sync-created, so Dawn defers their real compile to the forced `precompile`
    // dispatch and the drain times that (`Compute.precompiled`, whose JSDoc carries the why). What's
    // worth pinning here is that the hook lands in `Profile.compile` under the forcer's own label with
    // the drain's measured span — the punch item was typed pipelines vanishing from that table.
    test("Compute.precompiled records a typed pipeline's measured compile span", () => {
        const prev = Compute.device;
        Object.assign(Compute, { device: capableDevice() });
        try {
            ProfilePlugin.initialize?.(new State());
        } finally {
            Object.assign(Compute, { device: prev });
        }
        expect(Compute.precompiled).toBeDefined();
        Compute.precompiled?.("kitchen-part-count", 100, 137.5);
        expect(Profile.compile.get("kitchen-part-count")).toBe(37.5);
        // and it widens the startup span the same way an async pipeline's entry does
        Compute.precompiled?.("kitchen-part-scan", 137.5, 200);
        expect(Profile.compile.get("kitchen-part-scan")).toBe(62.5);
        expect(Profile.compileMs).toBeGreaterThanOrEqual(100);
    });
});
