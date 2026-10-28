import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { createApp, World } from "@dylanebert/shallot";
import { PhysicsMeshes } from "@dylanebert/shallot/physics";
import {
    createPhysicsMesh,
    destroyPhysicsMesh,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { countedKernelBytes } from "../../../scripts/counted-kernel";
import { meshBuild } from "./geometry-builds";
import { clockImport, type Kernel, kernelState } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);
let kernelBytes: ArrayBuffer;
beforeAll(async () => {
    kernelBytes = await countedKernelBytes();
}, TIMEOUT.startup);

const definition = {
    name: "mesh",
    grid: { xCount: 4, zCount: 5, cellWidth: 0.3, materialCount: 3, identifyEdges: true },
};

test("mesh release refuses foreign and released builds and invalid definitions", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const built = createPhysicsMesh(app.world, definition);
        expect(() => destroyPhysicsMesh(other.world, built)).toThrow("not live");
        destroyPhysicsMesh(app.world, built);
        expect(() => destroyPhysicsMesh(app.world, built)).toThrow("not live");
        expect(() =>
            createPhysicsMesh(app.world, {
                name: "invalid",
                vertices: new Float32Array(3),
                indices: new Uint32Array([0, 1, 2]),
            }),
        ).toThrow("past its");
    } finally {
        await other.dispose();
        await app.dispose();
    }
});

test("a released mesh report remains readable in full", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const report = createPhysicsMesh(app.world, definition);
        const expected = structuredClone(report);
        destroyPhysicsMesh(app.world, report);
        expect(report).toEqual(expected);
    } finally {
        await app.dispose();
    }
});

test("building after release returns a fresh report with equal data", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const first = createPhysicsMesh(app.world, definition);
        const expected = structuredClone(first);
        destroyPhysicsMesh(app.world, first);
        const rebuilt = createPhysicsMesh(app.world, definition);
        expect(rebuilt).not.toBe(first);
        expect(rebuilt).toEqual(expected);
        destroyPhysicsMesh(app.world, rebuilt);
    } finally {
        await app.dispose();
    }
});

test("explicit builds leave the registry's cached shape build unchanged", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const handle = app.world.resource(PhysicsMeshes).add(definition);
        const cached = meshBuild(app.world, handle)!;
        const before = JSON.stringify(cached);
        const first = createPhysicsMesh(app.world, definition);
        const second = createPhysicsMesh(app.world, definition);
        destroyPhysicsMesh(app.world, first);
        expect(meshBuild(app.world, handle)).toBe(cached);
        expect(JSON.stringify(cached)).toBe(before);
        expect(second.triangleCount).toBe(cached.triangles.length);
        destroyPhysicsMesh(app.world, second);
    } finally {
        await app.dispose();
    }
});

test("released reports and node bounds are collectable; the live control stays owned", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const finalized = new Set<string>();
    const registry = new FinalizationRegistry<string>((tag) => finalized.add(tag));
    const release = () => {
        const report = createPhysicsMesh(app.world, definition);
        registry.register(report, "report");
        registry.register(report.nodeBounds, "bounds");
        registry.register(report.nodes, "nodes");
        const refs = [
            new WeakRef(report),
            new WeakRef(report.nodeBounds),
            new WeakRef(report.nodes),
        ];
        destroyPhysicsMesh(app.world, report);
        return refs;
    };
    try {
        const refs = release();
        const control = createPhysicsMesh(app.world, definition);
        registry.register(control, "control", control);
        const deadline = Date.now() + 1000;
        while (Date.now() < deadline && finalized.size < 3) {
            await new Promise((resolve) => setTimeout(resolve, 0));
            Bun.gc(true);
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        expect(refs.every((ref) => ref.deref() === undefined)).toBe(true);
        expect(finalized).toEqual(new Set(["report", "bounds", "nodes"]));
        expect(registry.unregister(control)).toBe(true);
        destroyPhysicsMesh(app.world, control);
    } finally {
        await app.dispose();
    }
});

test("repeated mesh builds release all WASM bytes; an unreleased control retains bytes until World disposal", async () => {
    const built = await WebAssembly.instantiate(kernelBytes, {
        env: {
            now: clockImport(() => k.memory),
            queryCallback() {
                throw new Error("unexpected query");
            },
            materialCallback() {
                throw new Error("unexpected material callback");
            },
            collisionCallback() {
                throw new Error("unexpected collision callback");
            },
        },
    });
    const k = built.instance.exports as unknown as Kernel & { allocationLiveBytes(): number };
    const world = new World();
    kernelState(world).instance = k;
    for (let i = 0; i < 4; i++) destroyPhysicsMesh(world, createPhysicsMesh(world, definition));
    const baseline = k.allocationLiveBytes();
    const memory = k.memory.buffer.byteLength;
    for (let i = 0; i < 1000; i++) destroyPhysicsMesh(world, createPhysicsMesh(world, definition));
    expect(k.allocationLiveBytes()).toBe(baseline);
    expect(k.memory.buffer.byteLength).toBe(memory);
    createPhysicsMesh(world, definition);
    expect(k.allocationLiveBytes()).toBeGreaterThan(baseline);
    world.dispose();
    expect(k.allocationLiveBytes()).toBe(baseline);
});
