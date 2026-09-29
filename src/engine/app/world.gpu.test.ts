import { afterEach, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { attachCanvas, Camera, PointLight } from "../../core/rendering";
import { Sear, SearPlugin } from "../../standard/rendering";
import { Part, PartPlugin, Parts } from "../../transitional/part";
import { Body, PhysicsPlugin } from "../../transitional/physics";
import { SlabPlugin, slab } from "../../transitional/slab";
import { Transform } from "../../transitional/transforms";
import { Compute, f16x4, f32, field, type Plugin, type State, u8 } from "../index";
import { probeBuffer } from "../runtime";
import { serialize } from "../scene";
import { Xform } from "../utils";
import { build, swap } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
const createCanvasContext = (await import(peerModule)).GPUCanvasContextMock as new (
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
) => GPUCanvasContext;
await setupGlobals();
if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
        observe() {}
        disconnect() {}
    } as unknown as typeof ResizeObserver;
}

const Value = { amount: field(f32) };
const resourceKey = Symbol("world-probe");
const textureKey = Symbol("world-probe-texture");
const ResourcePlugin = {
    name: "WorldResourceProbe",
    initialize(state: State) {
        const buffer = state.resource(resourceKey, () =>
            Compute.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
        );
        const texture = state.resource(textureKey, () =>
            Compute.device.createTexture({
                size: [1, 1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            }),
        );
        const typed = Compute.root.createBuffer(d.arrayOf(d.u32, 1)).$usage("storage");
        const typedBuffer = Compute.root.unwrap(typed);
        Compute.buffers.set("world-probe", buffer);
        Compute.buffers.set("world-probe-typed", typedBuffer);
        Compute.textures.set("world-probe", texture);
        Compute.typed.set("world-probe-typed", typed);
    },
};
let apps: Awaited<ReturnType<typeof build>>[] = [];

afterEach(() => {
    for (const app of apps.splice(0)) app.dispose();
});

function amount(state: State) {
    return state.of(Value).amount;
}

function expectCapacityBuffers(state: State, capacity: number): void {
    expect(state.gpu.buffers.get("transforms")?.size).toBe(capacity * d.sizeOf(Xform));
    expect(state.gpu.buffers.get("membership")?.size).toBe(
        state.membership.generations * capacity * 4,
    );
    expect(state.gpu.buffers.get("material")?.size).toBe(capacity * f16x4.gpu!.bytes);
    expect(state.gpu.buffers.get("eids")?.size).toBe(capacity * 4);
}

function capacityViewPlugin(): Plugin {
    let context: GPUCanvasContext;
    const canvas = {
        width: 16,
        height: 16,
        style: { imageRendering: "auto" },
        getContext(kind: string) {
            return kind === "webgpu" ? context : null;
        },
        getBoundingClientRect() {
            return { width: 16, height: 16 };
        },
    } as unknown as HTMLCanvasElement;
    context = new createCanvasContext(canvas, canvas.width, canvas.height);
    return {
        name: "WorldCapacityView",
        dependencies: [SearPlugin],
        initialize(state) {
            const eid = state.create();
            state.add(eid, Transform);
            state.add(eid, Camera);
            state.add(eid, Sear);
            attachCanvas(eid, canvas, state);
        },
    };
}

test("live and later worlds keep component columns separate", async () => {
    const first = await build({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(first);
    const a = first.state.create();
    first.state.add(a, Value);
    amount(first.state).set(a, 11);

    const second = await build({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(second);
    const b = second.state.create();
    second.state.add(b, Value);
    expect(b).toBe(a);
    expect(amount(second.state).get(b)).toBe(0);
    amount(second.state).set(b, 22);
    expect(amount(first.state).get(a)).toBe(11);
    const firstBuffer = first.state.gpu.buffers.get("world-probe");
    const secondBuffer = second.state.gpu.buffers.get("world-probe");
    expect(firstBuffer).toBeDefined();
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(second.state.gpu.textures.get("world-probe")).not.toBe(
        first.state.gpu.textures.get("world-probe"),
    );
    expect(first.state.gpu.typed.get("world-probe-typed")).not.toBe(
        second.state.gpu.typed.get("world-probe-typed"),
    );

    first.dispose();
    apps = apps.filter((app) => app !== first);
    const later = await build({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(later);
    const c = later.state.create();
    later.state.add(c, Value);
    expect(c).toBe(a);
    expect(amount(later.state).get(c)).toBe(0);
    expect(later.state.gpu.buffers.get("world-probe")).not.toBe(secondBuffer);
    expect(later.state.gpu.textures.get("world-probe")).not.toBe(
        second.state.gpu.textures.get("world-probe"),
    );
    expect(later.state.gpu.typed.get("world-probe-typed")).not.toBe(
        second.state.gpu.typed.get("world-probe-typed"),
    );
});

test("component registrations, defaults, exclusions, and scene enumeration belong to each world", async () => {
    const firstComponents = {
        Value: { amount: field(f32) },
        Blocker: { value: field(f32) },
        Other: { value: field(f32) },
    };
    const secondComponents = {
        Value: { amount: field(f32), extra: field(f32) },
        Blocker: { value: field(f32) },
        Other: { value: field(f32) },
    };
    const firstPlugin = {
        name: "WorldRegistryProbe",
        components: firstComponents,
        traits: {
            Value: { defaults: () => ({ amount: 11 }) },
            Blocker: { excludes: [firstComponents.Other] },
        },
    };
    const secondPlugin = {
        name: "WorldRegistryProbe",
        components: secondComponents,
        traits: { Value: { defaults: () => ({ amount: 22 }) } },
    };
    const first = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(first);
    const second = await build({ defaults: false, plugins: [secondPlugin] });
    apps.push(second);

    const firstEid = first.state.create();
    const secondEid = second.state.create();
    first.state.add(firstEid, firstComponents.Value);
    second.state.add(secondEid, secondComponents.Value);
    expect(first.state.of(firstComponents.Value).amount.get(firstEid)).toBe(11);
    expect(second.state.of(secondComponents.Value).amount.get(secondEid)).toBe(22);

    first.state.add(firstEid, firstComponents.Blocker);
    expect(() => first.state.add(firstEid, firstComponents.Other)).toThrow('cannot attach "other"');
    second.state.add(secondEid, secondComponents.Blocker);
    expect(() => second.state.add(secondEid, secondComponents.Other)).not.toThrow();

    expect(
        serialize(first.state, [firstEid])[0]
            .attrs.map((attr) => attr.name)
            .sort(),
    ).toEqual(["blocker", "value"]);
}, 20_000);

test("world GPU registries and owned resources are isolated and released on dispose", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Dawn adapter unavailable");
    const device = await adapter.requestDevice();
    const liveBuffers = new Set<GPUBuffer>();
    const liveTextures = new Set<GPUTexture>();
    const wrappedDevice = new Proxy(device, {
        get(target, key) {
            if (key === "createBuffer") {
                return (descriptor: GPUBufferDescriptor) => {
                    const buffer = target.createBuffer(descriptor);
                    liveBuffers.add(buffer);
                    const destroy = buffer.destroy.bind(buffer);
                    buffer.destroy = () => {
                        if (liveBuffers.delete(buffer)) destroy();
                    };
                    return buffer;
                };
            }
            if (key === "createTexture") {
                return (descriptor: GPUTextureDescriptor) => {
                    const texture = target.createTexture(descriptor);
                    liveTextures.add(texture);
                    const destroy = texture.destroy.bind(texture);
                    texture.destroy = () => {
                        if (liveTextures.delete(texture)) destroy();
                    };
                    return texture;
                };
            }
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    const first = await build({
        defaults: false,
        plugins: [ResourcePlugin],
        device: wrappedDevice,
    });
    apps.push(first);
    const firstBuffer = first.state.gpu.buffers.get("world-probe");
    const firstTypedBuffer = first.state.gpu.buffers.get("world-probe-typed");
    const firstTexture = first.state.gpu.textures.get("world-probe");
    expect(firstBuffer).toBeDefined();
    expect(firstTypedBuffer).toBeDefined();
    expect(firstTexture).toBeDefined();

    const second = await build({
        defaults: false,
        plugins: [ResourcePlugin],
        device: wrappedDevice,
    });
    apps.push(second);
    const secondBuffer = second.state.gpu.buffers.get("world-probe");
    const secondTypedBuffer = second.state.gpu.buffers.get("world-probe-typed");
    const secondTexture = second.state.gpu.textures.get("world-probe");
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(secondTypedBuffer).toBeDefined();
    expect(secondTypedBuffer).not.toBe(firstTypedBuffer);
    expect(secondTexture).toBeDefined();
    expect(secondTexture).not.toBe(firstTexture);
    expect(first.state.gpu.buffers.get("world-probe")).toBe(firstBuffer);

    first.dispose();
    apps = apps.filter((app) => app !== first);
    expect(liveBuffers.has(firstBuffer!)).toBe(false);
    expect(liveBuffers.has(firstTypedBuffer!)).toBe(false);
    expect(liveBuffers.has(secondBuffer!)).toBe(true);
    expect(liveBuffers.has(secondTypedBuffer!)).toBe(true);
    expect(liveTextures.has(firstTexture!)).toBe(false);
    expect(liveTextures.has(secondTexture!)).toBe(true);
    second.dispose();
    apps = apps.filter((app) => app !== second);
    expect(liveBuffers.size).toBe(0);
    expect(liveTextures.size).toBe(0);
    device.destroy();
}, 20_000);

test("frame change marks clear after slab uploads, including fields without GPU uploads", async () => {
    const Changed = {
        sparse: field(f32),
        uploaded: slab(f32, "change-mark-uploaded"),
        unsupported: slab(u8),
    };
    const plugin = {
        name: "WorldChangeMarkProbe",
        dependencies: [SlabPlugin],
        components: { Changed },
    };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);
    const { state } = app;
    const eid = state.create();
    state.add(eid, Changed);
    const storage = state.of(Changed);
    storage.sparse.set(eid, 1);
    storage.uploaded.set(eid, 2);
    storage.unsupported.set(eid, 3);

    state.step(0);
    expect(
        [storage.sparse, storage.uploaded, storage.unsupported].map((field) =>
            Array.from(field.dirty).some((word) => word !== 0),
        ),
    ).toEqual([false, false, false]);

    let writeAfterUpload = false;
    const lateWriter = {
        group: "draw" as const,
        update(current: State) {
            if (writeAfterUpload) current.of(Changed).uploaded.set(eid, 9);
        },
    };
    state.addSystem(lateWriter);
    writeAfterUpload = true;
    state.step(0);
    expect(storage.uploaded.dirty[0]).not.toBe(0);

    writeAfterUpload = false;
    state.step(0);
    expect(storage.uploaded.dirty[0]).toBe(0);
}, 20_000);

test("each world's capacity sizes entities, membership, and Slab buffers independently", async () => {
    const CapacityField = { value: slab(f32, "world-capacity-value") };
    const plugin = {
        name: "WorldCapacityProbe",
        dependencies: [SlabPlugin],
        components: { CapacityField },
    };
    const first = await build({ defaults: false, plugins: [plugin], capacity: 64 });
    apps.push(first);
    const second = await build({ defaults: false, plugins: [plugin], capacity: 8 });
    apps.push(second);

    expect(first.state.capacity).toBe(64);
    expect(second.state.capacity).toBe(8);
    expect(first.state.gpu.buffers.get("world-capacity-value")?.size).toBe(64 * 4);
    expect(second.state.gpu.buffers.get("world-capacity-value")?.size).toBe(8 * 4);
    expect(first.state.gpu.buffers.get("membership")?.size).toBe(
        first.state.membership.generations * 64 * 4,
    );
    expect(second.state.gpu.buffers.get("membership")?.size).toBe(
        second.state.membership.generations * 8 * 4,
    );

    for (let i = 0; i < 63; i++) first.state.create();
    for (let i = 0; i < 7; i++) second.state.create();
    expect(() => first.state.create()).toThrow("configured capacity 64");
    expect(() => second.state.create()).toThrow("configured capacity 8");
}, 20_000);

test("built-in capacity readers size both worlds and render and step the top eid", async () => {
    const below = 32;
    const small = await build({
        defaults: false,
        plugins: [PhysicsPlugin, PartPlugin, SearPlugin, capacityViewPlugin()],
        capacity: below,
    });
    apps.push(small);
    expectCapacityBuffers(small.state, below);
    small.state.gpu.device.pushErrorScope("validation");
    small.state.step();
    await small.state.gpu.device.queue.onSubmittedWorkDone();
    expect(await small.state.gpu.device.popErrorScope()).toBeNull();
    small.dispose();
    apps = apps.filter((app) => app !== small);

    const above = 65544;
    const large = await build({
        defaults: false,
        plugins: [PhysicsPlugin, PartPlugin, SearPlugin, capacityViewPlugin()],
        capacity: above,
    });
    apps.push(large);
    expectCapacityBuffers(large.state, above);

    let topEid = 0;
    for (let i = 0; i < above - 2; i++) topEid = large.state.create();
    expect(topEid).toBe(above - 1);
    large.state.add(topEid, Body);
    large.state.add(topEid, Part);
    large.state.add(topEid, PointLight);

    let drawArgs: GPUBuffer | undefined;
    large.state.addSystem({
        name: "WorldCapacityDrawProbe",
        group: "draw",
        update() {
            if (!Parts.drawArgs) throw new Error("Part draw args were not allocated");
            drawArgs = Compute.root.unwrap(Parts.drawArgs);
        },
    });
    large.state.gpu.device.pushErrorScope("validation");
    large.state.step();
    await large.state.gpu.device.queue.onSubmittedWorkDone();
    expect(await large.state.gpu.device.popErrorScope()).toBeNull();

    const packed = large.state.gpu.buffers.get("eids");
    expect(packed).toBeDefined();
    expect(drawArgs).toBeDefined();
    const packedEid = await probeBuffer(large.state.gpu.device, packed!, { size: 4 });
    const drawInstanceCount = await probeBuffer(large.state.gpu.device, drawArgs!, {
        offset: 4,
        size: 4,
    });
    expect(new Uint32Array(packedEid.bytes)[0]).toBe(topEid);
    expect(new Uint32Array(drawInstanceCount.bytes)[0]).toBe(1);
}, 30_000);

test("reordered component fields swap without rebuilding their world columns", async () => {
    const firstValue = { x: field(f32), y: field(f32) };
    const firstPlugin = { name: "WorldFieldOrderProbe", components: { Value: firstValue } };
    const app = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);
    const eid = app.state.create();
    app.state.add(eid, firstValue);
    const before = app.state.of(firstValue);
    before.x.set(eid, 17);
    const beforeX = before.x.column;
    const beforeY = before.y.column;

    const reloadedValue = { y: field(f32), x: field(f32) };
    const reloaded = { name: "WorldFieldOrderProbe", components: { Value: reloadedValue } };
    expect(await swap(app.state, [firstPlugin], [reloaded])).toEqual({ ok: true });

    const after = app.state.of(reloadedValue);
    expect(after.x.column).toBe(beforeX);
    expect(after.y.column).toBe(beforeY);
    expect(after.x.get(eid)).toBe(17);
}, 20_000);

test("a same-named Type with a different array layout forces a rebuild", async () => {
    const firstValue = { amount: field(f32) };
    const firstPlugin = { name: "WorldTypeLayoutProbe", components: { Value: firstValue } };
    const app = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);

    const wordF32 = { ...f32, ctor: Uint32Array };
    const reloadedValue = { amount: field(wordF32) };
    const reloaded = {
        name: "WorldTypeLayoutProbe",
        components: { Value: reloadedValue },
    };
    expect(await swap(app.state, [firstPlugin], [reloaded])).toEqual({
        ok: false,
        reason: 'WorldTypeLayoutProbe: component "Value" schema changed',
    });
    app.state.registry.register("Value", reloadedValue);
    expect(() => app.state.of(reloadedValue)).toThrow("schema changed");
}, 20_000);

test("a Type's debug name does not invalidate an identical storage layout", async () => {
    const firstValue = { amount: field(f32) };
    const firstPlugin = { name: "WorldTypeDebugNameProbe", components: { Value: firstValue } };
    const app = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);

    const debugAlias = { ...f32, name: "f32-debug-alias" };
    const reloadedValue = { amount: field(debugAlias) };
    const reloaded = {
        name: "WorldTypeDebugNameProbe",
        components: { Value: reloadedValue },
    };
    expect(await swap(app.state, [firstPlugin], [reloaded])).toEqual({ ok: true });
}, 20_000);

test("original, reloaded, and rebuilt component accessors stop rechecking bound schemas", async () => {
    const originalValue = { amount: field(f32) };
    const originalPlugin = {
        name: "WorldAccessorCacheProbe",
        components: { Value: originalValue },
    };
    const first = await build({ defaults: false, plugins: [originalPlugin] });
    apps.push(first);
    const originalEid = first.state.create();
    first.state.add(originalEid, originalValue);

    const reloadedValue = { amount: field(f32) };
    const reloadedPlugin = {
        name: "WorldAccessorCacheProbe",
        components: { Value: reloadedValue },
    };
    expect(await swap(first.state, [originalPlugin], [reloadedPlugin])).toEqual({ ok: true });

    const rebuilt = await build({ defaults: false, plugins: [reloadedPlugin] });
    apps.push(rebuilt);
    const rebuiltEid = rebuilt.state.create();
    rebuilt.state.add(rebuiltEid, reloadedValue);

    const originalSort = Array.prototype.sort;
    let sortCalls = 0;
    let originalRead = 0;
    let reloadedRead = 0;
    let rebuiltRead = 0;
    Array.prototype.sort = function <T>(this: T[], compareFn?: (a: T, b: T) => number): T[] {
        sortCalls++;
        return originalSort.call(this, compareFn);
    };
    try {
        first.state.of(originalValue);
        originalValue.amount.set(originalEid, 11);
        originalRead = originalValue.amount.get(originalEid);

        first.state.of(reloadedValue);
        reloadedValue.amount.set(originalEid, 22);
        reloadedRead = reloadedValue.amount.get(originalEid);

        rebuilt.state.of(reloadedValue);
        reloadedValue.amount.set(rebuiltEid, 33);
        rebuiltRead = reloadedValue.amount.get(rebuiltEid);
    } finally {
        Array.prototype.sort = originalSort;
    }

    expect([originalRead, reloadedRead, rebuiltRead]).toEqual([11, 22, 33]);
    expect(sortCalls).toBe(0);
}, 20_000);

test("a compatible hot swap reattaches its schema in only the target world", async () => {
    const firstPlugin = {
        name: "SwappableWorldSchema",
        components: { Value },
        initialize(state: State) {
            const eid = state.create();
            state.add(eid, this.components.Value);
            amount(state).set(eid, 7);
        },
    };
    const first = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(first);
    const second = await build({ defaults: false, plugins: [firstPlugin] });
    apps.push(second);
    const firstEid = first.state.entities()[0];
    const secondEid = second.state.entities()[0];
    amount(first.state).set(firstEid, 13);

    const reloadedValue = { amount: field(f32) };
    const reloaded = {
        name: "SwappableWorldSchema",
        components: { Value: reloadedValue },
        initialize() {
            expect(reloadedValue.amount.get(firstEid)).toBe(13);
        },
    };
    expect(await swap(first.state, [firstPlugin], [reloaded])).toEqual({ ok: true });
    expect(amount(first.state).get(firstEid)).toBe(13);
    expect(amount(second.state).get(secondEid)).toBe(7);

    const incompatibleValue = { amount: field(f32), extra: field(f32) };
    const incompatible = {
        name: "SwappableWorldSchema",
        components: { Value: incompatibleValue },
    };
    expect(await swap(first.state, [reloaded], [incompatible])).toEqual({
        ok: false,
        reason: 'SwappableWorldSchema: component "Value" schema changed',
    });
    expect(amount(first.state).get(firstEid)).toBe(13);
    expect(amount(second.state).get(secondEid)).toBe(7);
}, 20_000);
