import { afterEach, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    createTestGpuApp as createApp,
    disposeGpuApps,
    expectAdapterNotice,
} from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { component } from "../ecs";
import { classifyAdapter } from "../runtime/adapter";

setDefaultTimeout(TIMEOUT.node);

import * as d from "typegpu/data";
import { Meshes } from "../../core/mesh";
import { Images, registerImage } from "../../core/rendering";
import { GlobalTransformHistory } from "../../core/rendering/global-transform";
import { GlobalTransform, Transform } from "../../core/transform";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "../../extras/lines";
import { segmentState } from "../../extras/lines/segments";
import { Sprite, SpritePlugin } from "../../extras/sprite";
import { SpriteSource } from "../../extras/sprite/material";
import {
    Content,
    Fonts,
    internText,
    registerFont,
    Text,
    TextPlugin,
    textStateKey,
} from "../../extras/text";
import { isolationFont } from "../../extras/text/font.fixture";
import { Draws } from "../../standard/rendering";
import { f32, probeBuffer, requestGPU, Time, World } from "../index";
import "../../standard";

import { setupGlobals } from "@dylanebert/shallot/webgpu";

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
const typedKey = {
    create: (world: World) => world.gpu.root.createBuffer(d.arrayOf(d.u32, 1)).$usage("storage"),
};
const ResourcePlugin = {
    name: "WorldResourceProbe",
    gpu: {},
    initialize(world: World) {
        world.resource(resourceKey);
        world.resource(textureKey);
        world.resource(typedKey);
    },
};
let apps: Awaited<ReturnType<typeof createApp>>[] = [];

afterEach(() => disposeGpuApps(apps.splice(0)));

function amount(world: World) {
    return world.storage(Value).amount;
}

const CompositionValue = { value: f32 };
const CompositionPlugin = {
    name: "InterleavedCompositionProbe",
    recovery: "stateless" as const,
    components: [component("CompositionValue", CompositionValue)],
    systems: [
        {
            group: "fixed" as const,
            update(world: World) {
                const values = world.storage(CompositionValue).value;
                for (const eid of world.query([CompositionValue])) {
                    values.set(eid, values.get(eid) + 1);
                }
            },
        },
    ],
    initialize(world: World) {
        const eid = world.create();
        world.add(eid, CompositionValue);
    },
};

test("CPU-only and default-rendered worlds step interleaved without sharing fields or acquiring for CPU", async () => {
    const gpu = navigator.gpu;
    let adapterRequests = 0;
    const trackedGpu = new Proxy(gpu, {
        get(target, key) {
            if (key === "requestAdapter") {
                return (...args: Parameters<GPU["requestAdapter"]>) => {
                    adapterRequests++;
                    return target.requestAdapter(...args);
                };
            }
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    const descriptor = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: trackedGpu });
    try {
        const cpu = await createApp({ defaults: false, plugins: [CompositionPlugin] });
        apps.push(cpu);
        expect(adapterRequests).toBe(0);
        expect(() => cpu.world.gpu).toThrow("no enabled plugin declares a GPU requirement");

        const rendered = await createApp({ plugins: [CompositionPlugin] });
        apps.push(rendered);
        expect(adapterRequests).toBe(1);
        expect(rendered.world.gpu).toBeDefined();

        cpu.world.step(Time.FIXED_DT);
        rendered.world.step(Time.FIXED_DT);
        cpu.world.step(Time.FIXED_DT);
        const cpuStorage = cpu.world.storage(CompositionValue).value;
        const renderedStorage = rendered.world.storage(CompositionValue).value;
        const cpuEid = [...cpu.world.query([CompositionValue])][0]!;
        const renderedEid = [...rendered.world.query([CompositionValue])][0]!;
        expect(cpuEid).toBe(renderedEid);
        expect(cpuStorage.column).not.toBe(renderedStorage.column);
        expect(cpuStorage.get(cpuEid)).toBe(2);
        expect(renderedStorage.get(renderedEid)).toBe(1);

        rendered.world.step(Time.FIXED_DT);
        expect(cpuStorage.get(cpuEid)).toBe(2);
        expect(renderedStorage.get(renderedEid)).toBe(2);
    } finally {
        if (descriptor) Object.defineProperty(navigator, "gpu", descriptor);
        else Reflect.deleteProperty(navigator, "gpu");
    }
});

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
    const firstBuffer = first.world.resource(resourceKey);
    const secondBuffer = second.world.resource(resourceKey);
    expect(firstBuffer).toBeDefined();
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(second.world.resource(textureKey)).not.toBe(first.world.resource(textureKey));
    expect(first.world.resource(typedKey)).not.toBe(second.world.resource(typedKey));

    await disposeGpuApps([first]);
    apps = apps.filter((app) => app !== first);
    const later = await createApp({ defaults: false, plugins: [ResourcePlugin] });
    apps.push(later);
    const c = later.world.create();
    later.world.add(c, Value);
    expect(c).toBe(a);
    expect(amount(later.world).get(c)).toBe(0);
    expect(later.world.resource(resourceKey)).not.toBe(secondBuffer);
    expect(later.world.resource(textureKey)).not.toBe(second.world.resource(textureKey));
    expect(later.world.resource(typedKey)).not.toBe(second.world.resource(typedKey));
});

test("component registrations and defaults belong to each world", async () => {
    const firstComponents = {
        Value: { amount: f32 },
    };
    const secondComponents = {
        Value: { amount: f32, extra: f32 },
    };
    const firstPlugin = {
        name: "WorldRegistryProbe",
        components: [
            component("Value", firstComponents.Value, { defaults: () => ({ amount: 11 }) }),
        ],
    };
    const secondPlugin = {
        name: "WorldRegistryProbe",
        components: [
            component("Value", secondComponents.Value, { defaults: () => ({ amount: 22 }) }),
        ],
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
});

test("world-owned GPU resources are isolated and released on dispose", async () => {
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
    const firstBuffer = first.world.resource(resourceKey);
    const firstTypedBuffer = first.world.gpu.root.unwrap(first.world.resource(typedKey));
    const firstTexture = first.world.resource(textureKey);
    expect(firstBuffer).toBeDefined();
    expect(firstTypedBuffer).toBeDefined();
    expect(firstTexture).toBeDefined();

    const second = await createApp({
        defaults: false,
        plugins: [ResourcePlugin],
        device: wrappedDevice,
    });
    apps.push(second);
    const secondBuffer = second.world.resource(resourceKey);
    const secondTypedBuffer = second.world.gpu.root.unwrap(second.world.resource(typedKey));
    const secondTexture = second.world.resource(textureKey);
    expect(secondBuffer).toBeDefined();
    expect(secondBuffer).not.toBe(firstBuffer);
    expect(secondTypedBuffer).toBeDefined();
    expect(secondTypedBuffer).not.toBe(firstTypedBuffer);
    expect(secondTexture).toBeDefined();
    expect(secondTexture).not.toBe(firstTexture);
    expect(first.world.resource(resourceKey)).toBe(firstBuffer);

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
    expectError(/GPU device lost destroyed: device was destroyed/);
    device.destroy();
});

test("entity ids and component columns grow without a configured capacity", async () => {
    const Grow = { value: f32 };
    const plugin = { name: "WorldGrowthProbe", components: [component("Grow", Grow)] };
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

const OWNERSHIP_FONT = `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`;

async function composition(seed: string, offset: number) {
    const app = await createApp({
        plugins: [TextPlugin, SpritePlugin, LinesPlugin],
        setup(world) {
            const font = registerFont(world, OWNERSHIP_FONT);
            const image = registerImage(world, new Blob([], { type: "image/png" }));
            const content = internText(world, seed === "first" ? "isolation" : "salt");
            const label = world.create();
            world.add(label, Transform, { translation: [offset, 1, 0, 0] });
            world.add(label, Text, { content, font });
            const sprite = world.create();
            world.add(sprite, Transform, { translation: [offset, 2, 0, 0] });
            world.add(sprite, Sprite, { image });
            const line = world.create();
            world.add(line, Transform, { translation: [offset, 0, 0, 0] });
            const asset = new GizmoAsset();
            drawLine(asset, [0, 0, 0], [1, 1, 1], 0xffffff);
            world.add(line, Gizmo, { handle: registerGizmo(world, asset) });
        },
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
    for (const [name, buffer] of [
        ["global-transform", world.resource(GlobalTransformHistory).current!.buffer],
        ["textGlyphs", world.gpu.root.unwrap(world.resource(textStateKey).glyphBuf!)],
        ["sprite-material", world.resource(SpriteSource).table.buffer],
        ["lineSegments", world.gpu.root.unwrap(segmentState(world).buffer!)],
    ] as const) {
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
    await disposeGpuApps([soloA]);
    const soloB = await composition("second", 30);
    const b = [];
    for (let i = 0; i < 3; i++) b.push(await compositionFrame(soloB.world, 30, i));
    await disposeGpuApps([soloB]);
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
        gpu: {},
        components: [component("Value", Value)],
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
        gpu: {},
        components: [component("Value", Value)],
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
            // A device without its adapter is stamped unidentified (requestGPU).
            expectAdapterNotice(classifyAdapter({ present: true }));
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
