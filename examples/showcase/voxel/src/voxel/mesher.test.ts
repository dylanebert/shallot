import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import type { State } from "@dylanebert/shallot";
import {
    body,
    flat,
    integerDiscipline,
    noDivision,
} from "../../../../../packages/shallot/tests/wgsl";
import { TOTAL_CELLS } from "./grid";
import {
    cleanupWarmOwner,
    commitEdit,
    createWarmLifecycle,
    disposeVoxelWarm,
    emitWgsl,
    uploadVoxels,
    VoxelEmitSystem,
    VoxelPlugin,
    Voxels,
    warmVoxelEmit,
} from "./mesher";

type FakeState = {
    disposed: boolean;
    onDispose: (cleanup: () => void) => void;
    dispose: () => void;
};

type FakeBuffer = { kind: "production" | "warm"; owner: number; destroyed: number };
type FakeBinding = { kind: "production" | "warm"; owner: number };
type FakeOwner = {
    state: FakeState;
    id: number;
    buffers: FakeBuffer[];
    cleaned: boolean;
};
type FakePrepared = { owner: FakeOwner; binding: FakeBinding };

function fakeState(disposed = false): FakeState {
    const cleanups: (() => void)[] = [];
    const state: FakeState = {
        disposed,
        onDispose(cleanup) {
            if (state.disposed) cleanup();
            else cleanups.push(cleanup);
        },
        dispose() {
            if (state.disposed) return;
            state.disposed = true;
            for (let i = cleanups.length - 1; i >= 0; i--) cleanups[i]();
        },
    };
    return state;
}

function deferred<T>() {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((yes, no) => {
        resolve = yes;
        reject = no;
    });
    return { promise, resolve, reject };
}

function fakeWarmHarness(deferredPrecompile = false) {
    let active: FakeOwner | null = null;
    let nextId = 0;
    let failPrepare = false;
    let failWarmBinding = false;
    const owners = new WeakMap<FakeState, FakeOwner>();
    const created: FakeOwner[] = [];
    const prepared: FakePrepared[] = [];
    const registered: number[] = [];
    const published: number[] = [];
    const forced: FakeBinding[] = [];
    const warmBuffers: FakeBuffer[] = [];
    const queued: {
        owner: FakeOwner;
        force: () => unknown;
        completion: ReturnType<typeof deferred<void>>;
    }[] = [];

    const cleanup = (owner: FakeOwner): void => {
        if (owner.cleaned) return;
        owner.cleaned = true;
        for (const buffer of owner.buffers) buffer.destroyed++;
        if (active === owner) active = null;
        if (owners.get(owner.state) === owner) owners.delete(owner.state);
    };

    const lifecycle = createWarmLifecycle({
        current: () => active,
        owned: (state: FakeState) => owners.get(state),
        create: (state: FakeState): FakeOwner => {
            const owner = { state, id: ++nextId, buffers: [], cleaned: false };
            created.push(owner);
            return owner;
        },
        activate(owner: FakeOwner) {
            active = owner;
            owners.set(owner.state, owner);
        },
        cleanup,
        prepare(owner: FakeOwner): FakePrepared {
            owner.buffers.push({ kind: "production", owner: owner.id, destroyed: 0 });
            if (failPrepare) throw new Error("partial allocation");
            registered.push(owner.id);
            const result = {
                owner,
                binding: { kind: "production" as const, owner: owner.id },
            };
            prepared.push(result);
            return result;
        },
        precompile(owner: FakeOwner, _prepared: FakePrepared, force: () => unknown) {
            const completion = deferred<void>();
            queued.push({ owner, force, completion });
            if (!deferredPrecompile) completion.resolve();
            return completion.promise;
        },
        createWarmBinding(own: (buffer: FakeBuffer) => void): FakeBinding {
            const buffer = { kind: "warm" as const, owner: 0, destroyed: 0 };
            warmBuffers.push(buffer);
            own(buffer);
            if (failWarmBinding) throw new Error("partial warm binding");
            return { kind: "warm", owner: 0 };
        },
        force(_prepared: FakePrepared, binding: FakeBinding) {
            forced.push(binding);
            return binding;
        },
        destroyWarm(buffer: FakeBuffer) {
            buffer.destroyed++;
        },
        publish(owner: FakeOwner) {
            published.push(owner.id);
        },
    });

    return {
        lifecycle,
        created,
        prepared,
        registered,
        published,
        forced,
        warmBuffers,
        queued,
        active: () => active,
        failNextPrepare: () => {
            failPrepare = true;
        },
        failNextWarmBinding: () => {
            failWarmBinding = true;
        },
    };
}

describe("voxel mesher emit kernel", () => {
    test("resolves the typed emit kernel and its shared helpers", () => {
        const wgsl = flat(emitWgsl());
        expect(wgsl).toContain("@compute @workgroup_size(4, 4, 4)");
        expect(wgsl).toContain("fn voxelemitkernel(");
        expect(wgsl).toContain("atomicAdd");
        expect(wgsl).toContain("voxelIndex");
        expect(wgsl).toContain("encodePos");
        expect(wgsl).toContain("encodeUv");
        const kernel = body(emitWgsl(), "@compute");
        integerDiscipline(kernel);
        noDivision(kernel);
        const hash = createHash("sha256").update(flat(kernel)).digest("hex");
        expect(hash).toBe("8fafc6a71ad0f3207505d88378368364ff636adaa20f21c5936c29fcfc417fe8");
        const mutated = flat(kernel).replace(
            "indices[(base + 0u)] = (v0 + 0u);",
            "indices[(base + 0u)] = (v0 + 1u);",
        );
        expect(createHash("sha256").update(mutated).digest("hex")).not.toBe(hash);
    });
});

describe("voxel mesher lifecycle", () => {
    test("keeps system setup synchronous and plugin warm owns the gpu prep", () => {
        expect(VoxelEmitSystem.setup).toBeDefined();
        expect(VoxelEmitSystem.setup?.constructor.name).not.toBe("AsyncFunction");
        expect(VoxelPlugin.warm).toBe(warmVoxelEmit);
        expect(VoxelPlugin.dispose).toBe(disposeVoxelWarm);
    });

    test("an already-disposed State leaves warm awaited and allocation-free", async () => {
        const state = { disposed: true } as unknown as State;
        const warming = VoxelPlugin.warm?.(state);
        expect(warming).toBeInstanceOf(Promise);
        await warming;
    });

    test("old cleanup preserves its replacement and active cleanup clears it once", () => {
        type Buffer = { destroyed: number };
        type Entry = { name: string; generation: number };
        type Owner = {
            readonly buffers: Buffer[];
            readonly meshes: Entry[];
            readonly draws: Entry[];
        };

        const oldBuffer = { destroyed: 0 };
        const nextBuffer = { destroyed: 0 };
        const oldMesh = { name: "voxel", generation: 1 };
        const nextMesh = { name: "voxel", generation: 2 };
        const oldDraw = { name: "voxel", generation: 1 };
        const nextDraw = { name: "voxel", generation: 2 };
        const oldOwner: Owner = {
            buffers: [oldBuffer],
            meshes: [oldMesh],
            draws: [oldDraw],
        };
        const nextOwner: Owner = {
            buffers: [nextBuffer],
            meshes: [nextMesh],
            draws: [nextDraw],
        };
        let mesh: Entry | undefined = nextMesh;
        let draw: Entry | undefined = nextDraw;
        let active: Owner | null = nextOwner;
        let globals: number | null = 2;
        const ops = {
            mesh: (name: string) => (mesh?.name === name ? mesh : undefined),
            draw: (name: string) => (draw?.name === name ? draw : undefined),
            deleteMesh: (name: string) => {
                if (mesh?.name === name) mesh = undefined;
            },
            deleteDraw: (name: string) => {
                if (draw?.name === name) draw = undefined;
            },
            destroy: (buffer: Buffer) => buffer.destroyed++,
            active: (owner: Owner) => active === owner,
            clearActive: () => {
                active = null;
                globals = null;
            },
        };

        cleanupWarmOwner(oldOwner, ops);
        expect(oldBuffer.destroyed).toBe(1);
        expect(mesh).toBe(nextMesh);
        expect(draw).toBe(nextDraw);
        expect(active).toBe(nextOwner);
        expect(globals).toBe(2);

        cleanupWarmOwner(nextOwner, ops);
        cleanupWarmOwner(nextOwner, ops);
        expect(nextBuffer.destroyed).toBe(1);
        expect(mesh).toBeUndefined();
        expect(draw).toBeUndefined();
        expect(active).toBeNull();
        expect(globals).toBeNull();
    });

    test("two generations queue throwaway forcers and old teardown preserves the replacement", async () => {
        const harness = fakeWarmHarness();
        const oldState = fakeState();
        const nextState = fakeState();

        await harness.lifecycle.warm(oldState);
        await harness.lifecycle.warm(nextState);
        const [oldOwner, nextOwner] = harness.created;
        expect(oldOwner.cleaned).toBe(true);
        expect(oldOwner.buffers[0].destroyed).toBe(1);
        expect(harness.active()).toBe(nextOwner);
        expect(harness.published).toEqual([oldOwner.id, nextOwner.id]);

        harness.queued[0].force();
        expect(harness.forced[0]).toEqual({ kind: "warm", owner: 0 });
        expect(harness.forced[0]).not.toBe(harness.prepared[0].binding);
        expect(harness.warmBuffers[0].destroyed).toBe(1);
        expect(oldOwner.buffers[0].destroyed).toBe(1);

        oldState.dispose();
        harness.lifecycle.dispose(oldState);
        expect(harness.active()).toBe(nextOwner);
        expect(nextOwner.cleaned).toBe(false);
    });

    test("an already-disposed State performs no lifecycle work", async () => {
        const harness = fakeWarmHarness();
        await harness.lifecycle.warm(fakeState(true));
        expect(harness.created).toHaveLength(0);
        expect(harness.registered).toHaveLength(0);
        expect(harness.queued).toHaveLength(0);
        expect(harness.published).toHaveLength(0);
    });

    test("partial allocation and stale precompile rejection clear only their generation", async () => {
        const partial = fakeWarmHarness();
        partial.failNextPrepare();
        await expect(partial.lifecycle.warm(fakeState())).rejects.toThrow("partial allocation");
        expect(partial.created[0].cleaned).toBe(true);
        expect(partial.created[0].buffers[0].destroyed).toBe(1);
        expect(partial.active()).toBeNull();
        expect(partial.registered).toHaveLength(0);

        const partialWarm = fakeWarmHarness();
        await partialWarm.lifecycle.warm(fakeState());
        partialWarm.failNextWarmBinding();
        expect(() => partialWarm.queued[0].force()).toThrow("partial warm binding");
        expect(partialWarm.warmBuffers[0].destroyed).toBe(1);
        expect(partialWarm.active()).toBe(partialWarm.created[0]);
        expect(partialWarm.created[0].cleaned).toBe(false);

        const harness = fakeWarmHarness(true);
        const oldWarm = harness.lifecycle.warm(fakeState());
        await Promise.resolve();
        const nextWarm = harness.lifecycle.warm(fakeState());
        await Promise.resolve();
        const [oldOwner, nextOwner] = harness.created;
        harness.queued[1].completion.resolve();
        await nextWarm;
        harness.queued[0].completion.reject(new Error("precompile rejected"));
        await expect(oldWarm).rejects.toThrow("precompile rejected");
        expect(oldOwner.cleaned).toBe(true);
        expect(harness.active()).toBe(nextOwner);
        expect(nextOwner.cleaned).toBe(false);
        expect(harness.published).toEqual([nextOwner.id]);
    });

    test("warm stays pending and unpublished until precompile completes", async () => {
        const harness = fakeWarmHarness(true);
        let resolved = false;
        const warming = harness.lifecycle.warm(fakeState()).then(() => {
            resolved = true;
        });
        await Promise.resolve();
        expect(harness.registered).toHaveLength(1);
        expect(harness.published).toHaveLength(0);
        expect(resolved).toBe(false);
        harness.queued[0].completion.resolve();
        await warming;
        expect(harness.published).toEqual([harness.created[0].id]);
        expect(resolved).toBe(true);
    });

    test("accepts authored voxels before setup without marking the mesher dirty", () => {
        const next = new Float32Array(TOTAL_CELLS);
        const prevData = Voxels.data;
        const prevDirty = Voxels.dirty;
        try {
            (Voxels as { data: Float32Array | null }).data = null;
            Voxels.dirty = false;
            uploadVoxels(next);
            expect(Voxels.data).toBe(next);
            expect(Voxels.dirty).toBe(false);
            expect(commitEdit([])).toBeUndefined();
        } finally {
            Voxels.data = prevData;
            Voxels.dirty = prevDirty;
        }
    });
});
