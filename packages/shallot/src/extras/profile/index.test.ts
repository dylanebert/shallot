import { describe, expect, test } from "bun:test";
import { Compute, State } from "../../engine";
import { UnsupportedError } from "../../engine/runtime";
import { Profile, ProfilePlugin } from "./index";
import { reorderRows } from "./reorder";

// `ProfilePlugin.features` declares `timestamp-query`, and `acquireDevice` throws on a device that can't
// grant it — but `requestGPU(externalDevice)` / `run({ device })` adopts a device as-is, so a declared
// feature never reaches an adopted one. `attach` is the layer both paths cross: it owns the engine's only
// `createQuerySet`, and without a guard there it would issue one against a device that can't validate it,
// leaving a raw GPU error on `onuncapturederror` and a profiler reporting silent zeros.
const device = (...features: string[]) =>
    ({ features: { has: (f: string) => features.includes(f) } }) as unknown as GPUDevice;

// a device that grants everything `attach` touches — enough to install the plugin's Compute sinks.
// Carries BOTH the async and sync pipeline constructors: TypeGPU only ever calls the sync pair
// (`createComputePipeline` / `createRenderPipeline`), so a mock missing them would make `attach`
// throw on `.bind` the moment it patches them, rather than exercising the patch.
const capableDevice = () =>
    ({
        features: { has: () => true },
        createQuerySet: () => ({}),
        createBuffer: () => ({ destroy() {} }),
        createTexture: () => ({ destroy() {} }),
        createComputePipelineAsync: async () => ({}),
        createRenderPipelineAsync: async () => ({}),
        createComputePipeline: () => ({}),
        createRenderPipeline: () => ({}),
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
    // tracking — and the pipeline-constructor patches below — on the FIRST device this module-lifetime
    // singleton attaches to. This test earns that slot by running first among the successful attaches in
    // this file — see the comment on the test below.
    //
    // The overlay's `collectStats` reads `profile.bufferBytes` / `profile.textureBytes` / `profile.sizes`
    // directly (index.ts `collectStats`) — the byte-budget gate's `Profile.bufferBytes` / `Profile.allocBytes`
    // must be that same field, not a parallel derivation the two could drift apart on. Pinning the public
    // seam against a live allocate + destroy is the differential.
    test("the public byte totals are the overlay's own fields, destroy() decrements them, and a sync-constructed pipeline registers in compile", () => {
        const prev = Compute.device;
        Object.assign(Compute, { device: capableDevice() });
        try {
            ProfilePlugin.initialize?.(new State());
            const buffer = Compute.device.createBuffer({
                label: "perf-gate-test",
                size: 1024,
                usage: GPUBufferUsage.STORAGE,
            });

            expect(Profile.bufferBytes).toBeGreaterThanOrEqual(1024);
            expect(Profile.allocBytes.get("perf-gate-test")).toBe(1024);

            const before = Profile.bufferBytes;
            buffer.destroy();
            expect(Profile.bufferBytes).toBe(before - 1024);
            expect(Profile.allocBytes.has("perf-gate-test")).toBe(false);

            // a `LazyAlloc.lazy: true` descriptor is a lazily-grown pool entry (`Mirror`'s readback
            // ring / `Slab`'s staging pool): its bytes sum into `lazyBytes`, a subset of `bufferBytes`,
            // so `bufferBytes - lazyBytes` (the byte-budget gate's exact total) doesn't move — a bogus
            // LAZY allocation must not red the gate. A bogus GATED allocation (no `lazy` mark) does move
            // it.
            expect(Profile.lazyBytes).toBe(0);
            const gatedBefore = Profile.bufferBytes - Profile.lazyBytes;
            const lazyBuffer = Compute.device.createBuffer({
                label: "perf-gate-lazy-test",
                size: 4096,
                usage: GPUBufferUsage.STORAGE,
                lazy: true,
            } as unknown as GPUBufferDescriptor);
            expect(Profile.lazyBytes).toBe(4096);
            expect(Profile.bufferBytes - Profile.lazyBytes).toBe(gatedBefore);
            const gatedBuffer = Compute.device.createBuffer({
                label: "perf-gate-gated-test",
                size: 256,
                usage: GPUBufferUsage.STORAGE,
            });
            expect(Profile.bufferBytes - Profile.lazyBytes).toBe(gatedBefore + 256);
            lazyBuffer.destroy();
            gatedBuffer.destroy();
            expect(Profile.lazyBytes).toBe(0);

            // TypeGPU builds every pipeline through the SYNCHRONOUS constructors
            // (`createRenderPipeline` / `createComputePipeline`), never the awaited `*Async` pair —
            // so a pipeline built this way must self-register in `compile` too, or `.size` stays a
            // count of hand-written `precompile` scope names rather than real pipelines.
            expect(Profile.compile.has("sync-render-pipeline")).toBe(false);
            Compute.device.createRenderPipeline({
                label: "sync-render-pipeline",
            } as unknown as GPURenderPipelineDescriptor);
            expect(Profile.compile.has("sync-render-pipeline")).toBe(true);

            expect(Profile.compile.has("sync-compute-pipeline")).toBe(false);
            Compute.device.createComputePipeline({
                label: "sync-compute-pipeline",
            } as unknown as GPUComputePipelineDescriptor);
            expect(Profile.compile.has("sync-compute-pipeline")).toBe(true);

            // both came from a real constructor call, so both count toward the pipeline-count golden
            // (`Profile.compiledPipelines`), not just `compile`'s raw key set — see the review test below,
            // which pins the opposite case (a scope-only label that never constructed a pipeline).
            expect(Profile.compiledPipelines.has("sync-render-pipeline")).toBe(true);
            expect(Profile.compiledPipelines.has("sync-compute-pipeline")).toBe(true);

            // a missing label carries no naming discipline — two anonymous pipelines are genuinely
            // distinct, so collapsing them would silently undercount every one past the first.
            Compute.device.createRenderPipeline({} as unknown as GPURenderPipelineDescriptor);
            Compute.device.createRenderPipeline({} as unknown as GPURenderPipelineDescriptor);
            const unlabeled = [...Profile.compile.keys()].filter((k) =>
                k.startsWith("(unlabeled)"),
            );
            expect(unlabeled.length).toBe(2);

            // a NAMED label is a stable per-pipeline diagnostic key (gpu.md), so a repeat overwrites —
            // the same convergence a typed pipeline's sync-constructor stub and its precompile forcer's
            // later, real completion measurement rely on to land under one label, not two.
            const callsBefore = Profile.pipelineCalls;
            const distinctBefore = Profile.compiledPipelines.size;
            Compute.device.createRenderPipeline({
                label: "sync-render-pipeline",
            } as unknown as GPURenderPipelineDescriptor);
            const collided = [...Profile.compile.keys()].filter((k) =>
                k.startsWith("sync-render-pipeline"),
            );
            expect(collided.length).toBe(1);

            // …which is exactly why the byte/label goldens gained a raw-call axis beside them: the second
            // pipeline is real but the distinct-label count didn't move, so that count alone cannot see a
            // pipeline multiplied under an existing label. `pipelineCalls` counts the call itself, so the
            // gym's `budget:pipeline-calls` golden moves where `budget:pipelines` can't
            // (`examples/gym/src/scenarios/budgets.ts`).
            expect(Profile.pipelineCalls).toBe(callsBefore + 1);
            expect(Profile.compiledPipelines.size).toBe(distinctBefore);
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
        Compute.precompiled?.("shallot-part-count", 100, 137.5);
        expect(Profile.compile.get("shallot-part-count")).toBe(37.5);
        // and it widens the startup span the same way an async pipeline's entry does
        Compute.precompiled?.("shallot-part-scan", 137.5, 200);
        expect(Profile.compile.get("shallot-part-scan")).toBe(62.5);
        expect(Profile.compileMs).toBeGreaterThanOrEqual(100);
    });

    // `precompileVariants` wraps N real pipeline compiles in ONE `Compute.precompiled` call under a
    // scope label that matches no actual pipeline (`"sear-typed-variants"`) — its timing is real and
    // belongs in `compile`, but counting it as a pipeline is exactly the blind spot closed for
    // per-variant pipelines and must not reopen for the scope label itself.
    test("a scope-only Compute.precompiled label lands in compile but not compiledPipelines", () => {
        const prev = Compute.device;
        Object.assign(Compute, { device: capableDevice() });
        try {
            ProfilePlugin.initialize?.(new State());
        } finally {
            Object.assign(Compute, { device: prev });
        }
        Compute.precompiled?.("sear-typed-variants", 0, 0.16);
        expect(Profile.compile.get("sear-typed-variants")).toBe(0.16);
        expect(Profile.compiledPipelines.has("sear-typed-variants")).toBe(false);
    });
});

describe("reorderRows", () => {
    // The hysteresis reorder splices `next` in place but re-inserts the element
    // from the *original* array at index i (`order[i]`) instead of the element
    // it actually removed. After the first move, `next` and `order` diverge —
    // `next[i]` differs from `order[i]` — so a later move re-inserts the
    // wrong name: the removed name is dropped and `order[i]` is duplicated.
    //
    // order = [a, b, c, d, e], rank = {b:0, d:1, e:2, a:3, c:4}
    //   move 1 (i=0): removes "a", re-inserts order[0]="a" at 3 → [b,c,d,a,e]
    //   move 2 (i=1): next[1]="c" ≠ order[1]="b"; removes "c", re-inserts
    //                 order[1]="b" at 4 → [b,d,a,e,b]
    //   move 3 (i=4): next[4]="b" ≠ order[4]="e"; removes "b", re-inserts
    //                 order[4]="e" at 0 → [e,b,d,a,e]
    // Pre-fix witness: output was ["e","b","d","a","e"] — "e" duplicated, "c" dropped.
    test("a permutation whose second move exposes prev[i] !== next[i] yields a permutation of the input", () => {
        const order = ["a", "b", "c", "d", "e"];
        const rank = new Map<string, number>([
            ["b", 0],
            ["d", 1],
            ["e", 2],
            ["a", 3],
            ["c", 4],
        ]);
        const result = reorderRows(order, rank);
        // same multiset — no duplicate, no drop
        expect(result.slice().sort()).toEqual(order.slice().sort());
    });
});
