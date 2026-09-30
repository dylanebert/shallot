import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import * as d from "typegpu/data";
import { Draws, Meshes } from "../../core/rendering";
import { drawLine, LinesPlugin } from "../../extras/lines";
import { Images, registerImage, Sprite, SpritePlugin } from "../../extras/sprite";
import { Content, Fonts, internText, registerFont, Text, TextPlugin } from "../../extras/text";
import { isolationFont } from "../../extras/text/font.fixture";
import { f32, GlobalTransform, probeBuffer, requestGPU, Time, Transform, World } from "../index";
import "../../standard";
import { serializeScene } from "../scene";
import { createApp, swapPlugins } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const Value = { amount: f32 };
const resourceKey = {
    create: (world: World) =>
        world.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
};
const textureKey = {
    create: (world: World) =>
        world.gpu.device.createTexture({
            size: [1, 1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        }),
};
const ResourcePlugin = {
    name: "WorldResourceProbe",
    initialize(world: World) {
        const buffer = world.resource(resourceKey);
        const texture = world.resource(textureKey);
        const typed = world.gpu.root.createBuffer(d.arrayOf(d.u32, 1)).$usage("storage");
        const typedBuffer = world.gpu.root.unwrap(typed);
        world.gpu.buffers.set("world-probe", buffer);
        world.gpu.buffers.set("world-probe-typed", typedBuffer);
        world.gpu.textures.set("world-probe", texture);
        world.gpu.typed.set("world-probe-typed", typed);
    },
};
let apps: Awaited<ReturnType<typeof createApp>>[] = [];

afterEach(() => {
    for (const app of apps.splice(0)) app.dispose();
});

function amount(world: World) {
    return world.storage(Value).amount;
}

test("live and later worlds keep component columns separate", async () => {
    const first = await createApp({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(first);
    const a = first.world.create();
    first.world.add(a, Value);
    amount(first.world).set(a, 11);

    const second = await createApp({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(second);
    const b = second.world.create();
    second.world.add(b, Value);
    expect(b).toBe(a);
    expect(amount(second.world).get(b)).toBe(0);
    amount(second.world).set(b, 22);
    expect(amount(first.world).get(a)).toBe(11);
    const firstBuffer = first.world.gpu.buffers.get("world-probe");
    const secondBuffer = second.world.gpu.buffers.get("world-probe");
    expect(firstBuffer).toBeDefined();
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(second.world.gpu.textures.get("world-probe")).not.toBe(
        first.world.gpu.textures.get("world-probe"),
    );
    expect(first.world.gpu.typed.get("world-probe-typed")).not.toBe(
        second.world.gpu.typed.get("world-probe-typed"),
    );

    first.dispose();
    apps = apps.filter((app) => app !== first);
    const later = await createApp({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(later);
    const c = later.world.create();
    later.world.add(c, Value);
    expect(c).toBe(a);
    expect(amount(later.world).get(c)).toBe(0);
    expect(later.world.gpu.buffers.get("world-probe")).not.toBe(secondBuffer);
    expect(later.world.gpu.textures.get("world-probe")).not.toBe(
        second.world.gpu.textures.get("world-probe"),
    );
    expect(later.world.gpu.typed.get("world-probe-typed")).not.toBe(
        second.world.gpu.typed.get("world-probe-typed"),
    );
});

test("component registrations, defaults, exclusions, and scene enumeration belong to each world", async () => {
    const firstComponents = {
        Value: { amount: f32 },
        Blocker: { value: f32 },
        Other: { value: f32 },
    };
    const secondComponents = {
        Value: { amount: f32, extra: f32 },
        Blocker: { value: f32 },
        Other: { value: f32 },
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
    const first = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(first);
    const second = await createApp({ defaults: false, plugins: [secondPlugin] });
    apps.push(second);

    const firstEid = first.world.create();
    const secondEid = second.world.create();
    first.world.add(firstEid, firstComponents.Value);
    second.world.add(secondEid, secondComponents.Value);
    expect(first.world.storage(firstComponents.Value).amount.get(firstEid)).toBe(11);
    expect(second.world.storage(secondComponents.Value).amount.get(secondEid)).toBe(22);

    first.world.add(firstEid, firstComponents.Blocker);
    expect(() => first.world.add(firstEid, firstComponents.Other)).toThrow('cannot attach "other"');
    second.world.add(secondEid, secondComponents.Blocker);
    expect(() => second.world.add(secondEid, secondComponents.Other)).not.toThrow();

    expect(
        serializeScene(first.world, [firstEid])[0]
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
    const first = await createApp({
        defaults: false,
        plugins: [ResourcePlugin],
        device: wrappedDevice,
    });
    apps.push(first);
    const firstBuffer = first.world.gpu.buffers.get("world-probe");
    const firstTypedBuffer = first.world.gpu.buffers.get("world-probe-typed");
    const firstTexture = first.world.gpu.textures.get("world-probe");
    expect(firstBuffer).toBeDefined();
    expect(firstTypedBuffer).toBeDefined();
    expect(firstTexture).toBeDefined();

    const second = await createApp({
        defaults: false,
        plugins: [ResourcePlugin],
        device: wrappedDevice,
    });
    apps.push(second);
    const secondBuffer = second.world.gpu.buffers.get("world-probe");
    const secondTypedBuffer = second.world.gpu.buffers.get("world-probe-typed");
    const secondTexture = second.world.gpu.textures.get("world-probe");
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(secondTypedBuffer).toBeDefined();
    expect(secondTypedBuffer).not.toBe(firstTypedBuffer);
    expect(secondTexture).toBeDefined();
    expect(secondTexture).not.toBe(firstTexture);
    expect(first.world.gpu.buffers.get("world-probe")).toBe(firstBuffer);

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
    const Changed = { sparse: f32, uploaded: f32 };
    const plugin = { name: "WorldChangeMarkProbe", components: { Changed } };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    apps.push(app);
    const { world } = app;
    const eid = world.create();
    world.add(eid, Changed);
    const storage = world.storage(Changed);
    storage.sparse.set(eid, 1);
    storage.uploaded.set(eid, 2);

    world.step(0);
    expect(
        ["sparse", "uploaded"].map((name) =>
            Array.from(world.fieldStorage(Changed, name).dirty).some((word) => word !== 0),
        ),
    ).toEqual([false, false]);

    let writeAfterUpload = false;
    const lateWriter = {
        group: "draw" as const,
        update(current: World) {
            if (writeAfterUpload) current.storage(Changed).uploaded.set(eid, 9);
        },
    };
    world.addSystem(lateWriter);
    writeAfterUpload = true;
    world.step(0);
    expect(world.fieldStorage(Changed, "uploaded").dirty[0]).not.toBe(0);

    writeAfterUpload = false;
    world.step(0);
    expect(world.fieldStorage(Changed, "uploaded").dirty[0]).toBe(0);
});

test("entity ids and component columns grow without a configured capacity", async () => {
    const Grow = { value: f32 };
    const plugin = { name: "WorldGrowthProbe", components: { Grow } };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    apps.push(app);
    let eid = 0;
    for (let i = 0; i < 4096; i++) eid = app.world.create();
    app.world.add(eid, Grow);
    app.world.storage(Grow).value.set(eid, 73.5);
    expect(app.world.entityHighWater).toBe(eid + 1);
    expect(app.world.storage(Grow).value.column.length).toBeGreaterThan(eid);
    expect(app.world.storage(Grow).value.get(eid)).toBe(73.5);
});

test("reordered component fields swap without rebuilding their world columns", async () => {
    const firstValue = { x: f32, y: f32 };
    const firstPlugin = { name: "WorldFieldOrderProbe", components: { Value: firstValue } };
    const app = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);
    const eid = app.world.create();
    app.world.add(eid, firstValue);
    const before = app.world.storage(firstValue);
    before.x.set(eid, 17);
    const beforeX = before.x.column;
    const beforeY = before.y.column;

    const reloadedValue = { y: f32, x: f32 };
    const reloaded = { name: "WorldFieldOrderProbe", components: { Value: reloadedValue } };
    expect(await swapPlugins(app.world, [firstPlugin], [reloaded])).toEqual({ ok: true });

    const after = app.world.storage(reloadedValue);
    expect(after.x.column).toBe(beforeX);
    expect(after.y.column).toBe(beforeY);
    expect(after.x.get(eid)).toBe(17);
});

test("a same-named Type with a different array layout forces a rebuild", async () => {
    const firstValue = { amount: f32 };
    const firstPlugin = { name: "WorldTypeLayoutProbe", components: { Value: firstValue } };
    const app = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);

    const wordF32 = { ...f32, ctor: Uint32Array };
    const reloadedValue = { amount: wordF32 };
    const reloaded = {
        name: "WorldTypeLayoutProbe",
        components: { Value: reloadedValue },
    };
    expect(await swapPlugins(app.world, [firstPlugin], [reloaded])).toEqual({
        ok: false,
        reason: 'WorldTypeLayoutProbe: component "Value" schema changed',
    });
    app.world.registry.register("Value", reloadedValue);
    expect(() => app.world.storage(reloadedValue)).toThrow("schema changed");
});

test("a Type's debug name does not invalidate an identical storage layout", async () => {
    const firstValue = { amount: f32 };
    const firstPlugin = { name: "WorldTypeDebugNameProbe", components: { Value: firstValue } };
    const app = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(app);

    const debugAlias = { ...f32, name: "f32-debug-alias" };
    const reloadedValue = { amount: debugAlias };
    const reloaded = {
        name: "WorldTypeDebugNameProbe",
        components: { Value: reloadedValue },
    };
    expect(await swapPlugins(app.world, [firstPlugin], [reloaded])).toEqual({ ok: true });
});

test("original, reloaded, and rebuilt component accessors stop rechecking bound schemas", async () => {
    const originalValue = { amount: f32 };
    const originalPlugin = {
        name: "WorldAccessorCacheProbe",
        components: { Value: originalValue },
    };
    const first = await createApp({ defaults: false, plugins: [originalPlugin] });
    apps.push(first);
    const originalEid = first.world.create();
    first.world.add(originalEid, originalValue);

    const reloadedValue = { amount: f32 };
    const reloadedPlugin = {
        name: "WorldAccessorCacheProbe",
        components: { Value: reloadedValue },
    };
    expect(await swapPlugins(first.world, [originalPlugin], [reloadedPlugin])).toEqual({
        ok: true,
    });

    const rebuilt = await createApp({ defaults: false, plugins: [reloadedPlugin] });
    apps.push(rebuilt);
    const rebuiltEid = rebuilt.world.create();
    rebuilt.world.add(rebuiltEid, reloadedValue);

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
        const original = first.world.storage(originalValue);
        original.amount.set(originalEid, 11);
        originalRead = original.amount.get(originalEid);

        const reloaded = first.world.storage(reloadedValue);
        reloaded.amount.set(originalEid, 22);
        reloadedRead = reloaded.amount.get(originalEid);

        const rebuiltStorage = rebuilt.world.storage(reloadedValue);
        rebuiltStorage.amount.set(rebuiltEid, 33);
        rebuiltRead = rebuiltStorage.amount.get(rebuiltEid);
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
        initialize(world: World) {
            const eid = world.create();
            world.add(eid, this.components.Value);
            amount(world).set(eid, 7);
        },
    };
    const first = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(first);
    const second = await createApp({ defaults: false, plugins: [firstPlugin] });
    apps.push(second);
    const firstEid = first.world.entities()[0];
    const secondEid = second.world.entities()[0];
    amount(first.world).set(firstEid, 13);

    const reloadedValue = { amount: f32 };
    const reloaded = {
        name: "SwappableWorldSchema",
        components: { Value: reloadedValue },
        initialize(world: World) {
            expect(world.storage(reloadedValue).amount.get(firstEid)).toBe(13);
        },
    };
    expect(await swapPlugins(first.world, [firstPlugin], [reloaded])).toEqual({ ok: true });
    expect(amount(first.world).get(firstEid)).toBe(13);
    expect(amount(second.world).get(secondEid)).toBe(7);

    const incompatibleValue = { amount: f32, extra: f32 };
    const incompatible = {
        name: "SwappableWorldSchema",
        components: { Value: incompatibleValue },
    };
    expect(await swapPlugins(first.world, [reloaded], [incompatible])).toEqual({
        ok: false,
        reason: 'SwappableWorldSchema: component "Value" schema changed',
    });
    expect(amount(first.world).get(firstEid)).toBe(13);
    expect(amount(second.world).get(secondEid)).toBe(7);
});

const OWNERSHIP_FONT = `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`;

async function composition(seed: string, offset: number) {
    const app = await createApp({
        plugins: [TextPlugin, SpritePlugin, LinesPlugin],
        setup(world) {
            registerFont(world, OWNERSHIP_FONT, seed);
            registerImage(world, new Blob([], { type: "image/png" }), seed);
            internText(world, seed === "first" ? "isolation" : "salt");
        },
        scene: `<scene>
            <a id="label" transform="translation: ${offset} 1 0" text="content: ${seed === "first" ? "isolation" : "salt"}; font: ${seed}" />
            <a id="sprite" transform="translation: ${offset} 2 0" sprite="image: ${seed}" />
            <a id="line" transform="translation: ${offset} 0 0" line="offset: 1 1 1" />
        </scene>`,
    });
    apps.push(app);
    console.info(
        `ownership ${seed} adapter: ${app.world.gpu.adapter.class} (${app.world.gpu.adapter.identity})`,
    );
    return app;
}

async function compositionFrame(world: World, offset: number, frame: number) {
    const transform = world.storage(Transform);
    for (const eid of world.query([Transform])) transform.translation.x.set(eid, offset + frame);
    drawLine(world, [offset, frame, 0], [offset + 1, frame + 1, 1], 0xffcc44);
    world.step(Time.FIXED_DT);
    const fields = [...world.query([Transform])].map((eid) => ({
        eid,
        pos: Array.from(transform.translation.read(eid, new Float32Array(4))),
        global: Array.from(
            world.storage(GlobalTransform).translation.read(eid, new Float32Array(4)),
        ),
        text: world.has(eid, Text) ? world.storage(Text).content.get(eid) : null,
        sprite: world.has(eid, Sprite) ? world.storage(Sprite).image.get(eid) : null,
    }));
    const buffers: Record<string, number[]> = {};
    for (const name of ["global-transform", "textGlyphs", "spriteData", "lineSegments"]) {
        const buffer = world.gpu.buffers.get(name);
        if (!buffer) throw new Error(`ownership composition did not publish ${name}`);
        const probe = await probeBuffer(world, buffer, { size: Math.min(buffer.size, 2048) });
        buffers[name] = Array.from(new Uint8Array(probe.bytes));
    }
    return {
        fields,
        fonts: [...world.resource(Fonts)].map((entry) => entry.name),
        content: [...world.resource(Content)].map((entry) => entry.name),
        images: [...world.resource(Images)].map((entry) => entry.name),
        meshes: [...world.resource(Meshes)].map((entry) => entry.name),
        draws: [...world.resource(Draws)].map((entry) => entry.name),
        buffers,
    };
}

test("interleaved default worlds with text, sprite and lines equal each world stepped alone in fields, names and GPU bytes", async () => {
    const soloA = await composition("first", 10);
    const a = [];
    for (let i = 0; i < 3; i++) a.push(await compositionFrame(soloA.world, 10, i));
    soloA.dispose();
    const soloB = await composition("second", 30);
    const b = [];
    for (let i = 0; i < 3; i++) b.push(await compositionFrame(soloB.world, 30, i));
    soloB.dispose();
    const first = await composition("first", 10);
    const second = await composition("second", 30);
    for (let i = 0; i < 3; i++) {
        second.world.storage(Transform);
        expect(await compositionFrame(first.world, 10, i)).toEqual(a[i]);
        first.world.storage(Transform);
        expect(await compositionFrame(second.world, 30, i)).toEqual(b[i]);
    }
});

test("nested and asynchronous lifecycle hooks retain explicit field, resource and GPU ownership", async () => {
    const declaration = { create: () => ({ value: 0 }) };
    let parent: World | undefined;
    const childPlugin = {
        name: "ExplicitChild",
        components: { Value },
        async initialize(world: World) {
            const _declaration = world.resource(declaration);

            const eid = world.create();
            world.add(eid, Value);
            world.storage(Value).amount.set(eid, 22);
            _declaration.value = 22;
            await Promise.resolve();
            parent!.storage(Value);
            expect(world.storage(Value).amount.get(eid)).toBe(22);
            expect(_declaration.value).toBe(22);
            expect(world.gpu).not.toBe(parent!.gpu);
        },
    };
    const parentPlugin = {
        name: "ExplicitParent",
        components: { Value },
        async warm(world: World) {
            const _declaration = world.resource(declaration);

            parent = world;
            const eid = world.create();
            world.add(eid, Value);
            world.storage(Value).amount.set(eid, 11);
            _declaration.value = 11;
            // Nested build waits for the build lock; direct lifecycle invocation exercises nesting
            // without asking that serialization contract to become reentrant.
            const child = new World();
            child.attachGpu(await requestGPU(parent!.gpu.device));
            try {
                await childPlugin.initialize(child);
                await Promise.resolve();
                child.storage(Value);
                expect(world.storage(Value).amount.get(eid)).toBe(11);
                expect(_declaration.value).toBe(11);
            } finally {
                child.dispose();
            }
        },
    };
    const app = await createApp({ defaults: false, plugins: [parentPlugin] });
    apps.push(app);
});
