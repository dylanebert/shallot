import { afterEach, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { Compute, f32, type State, sparse } from "../index";
import { build, swap } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const Value = { amount: sparse(f32) };
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

    const reloadedValue = { amount: sparse(f32) };
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

    const incompatibleValue = { amount: sparse(f32), extra: sparse(f32) };
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
