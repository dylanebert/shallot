import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import * as d from "typegpu/data";
import { Compute, f32, field, type State } from "../index";
import { serialize } from "../scene";
import { build, swap } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const Value = { amount: field(f32) };
const resourceKey = {
    create: () => Compute.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
};
const textureKey = {
    create: () =>
        Compute.device.createTexture({
            size: [1, 1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        }),
};
const ResourcePlugin = {
    name: "WorldResourceProbe",
    initialize(state: State) {
        const buffer = state.resource(resourceKey);
        const texture = state.resource(textureKey);
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
});

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
});

test("frame change marks clear at the world upload point", async () => {
    const Changed = { sparse: field(f32), uploaded: field(f32) };
    const plugin = { name: "WorldChangeMarkProbe", components: { Changed } };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);
    const { state } = app;
    const eid = state.create();
    state.add(eid, Changed);
    const storage = state.of(Changed);
    storage.sparse.set(eid, 1);
    storage.uploaded.set(eid, 2);

    state.step(0);
    expect(
        ["sparse", "uploaded"].map((name) =>
            Array.from(state.fieldStorage(Changed, name).dirty).some((word) => word !== 0),
        ),
    ).toEqual([false, false]);

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
    expect(state.fieldStorage(Changed, "uploaded").dirty[0]).not.toBe(0);

    writeAfterUpload = false;
    state.step(0);
    expect(state.fieldStorage(Changed, "uploaded").dirty[0]).toBe(0);
});

test("entity ids and component columns grow without a configured capacity", async () => {
    const Grow = { value: field(f32) };
    const plugin = { name: "WorldGrowthProbe", components: { Grow } };
    const app = await build({ defaults: false, plugins: [plugin] });
    apps.push(app);
    let eid = 0;
    for (let i = 0; i < 4096; i++) eid = app.state.create();
    app.state.add(eid, Grow);
    Grow.value.set(eid, 73.5);
    expect(app.state.entityHighWater).toBe(eid + 1);
    expect(app.state.of(Grow).value.column.length).toBeGreaterThan(eid);
    expect(Grow.value.get(eid)).toBe(73.5);
});

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
});

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
});

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
});

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
});

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
});
