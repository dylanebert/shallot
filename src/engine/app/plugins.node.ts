import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { MeshInstance } from "../../core/mesh";
import { Body, BodyType, DistanceJoint, ShapeKind, SphericalJoint } from "../../core/physics";
import {
    AmbientLight,
    attachCanvas,
    Camera,
    DepthPrepass,
    DirectionalLight,
    RenderingPlugin,
    Resolution,
    SpotLight,
    Tonemapping,
    Views,
    VolumetricLight,
} from "../../core/rendering";
import { offscreenTexture } from "../../core/rendering/view";
import { GlobalTransform } from "../../core/transform";
import {
    Arrow,
    Fog,
    FogPlugin,
    internText,
    Line,
    LinesPlugin,
    Orbit,
    OrbitPlugin,
    Outline,
    OutlinePlugin,
    Player,
    PlayerPlugin,
    ProfilePlugin,
    registerFont,
    Sky,
    SkyPlugin,
    Sprite,
    SpritePlugin,
    Text,
    TextPlugin,
    Vignette,
    VignettePlugin,
} from "../../extras";
import { isolationFont } from "../../extras/text/font.fixture";
import {
    DEFAULT_PLUGINS,
    Materials,
    MeshMaterial,
    StandardMaterial,
    Transform,
} from "../../standard";
import {
    Character,
    CharacterPlugin,
    hashPhysics,
    physicsWorld,
    readBody,
    StandardPhysicsPlugin,
    setVelocity,
} from "../../standard/physics";
import {
    Backgrounds,
    CameraBackground,
    cascadeComboEids,
    pointComboEids,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { AudioPlugin, Listener, Sound } from "../../transitional/audio";
import { type Bvh, BvhPlugin, createBvh } from "../../transitional/bvh";
import { type Plugin, probeTexture, Time, type World } from "../index";
import { CanvasContext } from "./canvas.fixture";
import { createApp } from "./index";

const everyPlugin: readonly Plugin[] = [
    ...DEFAULT_PLUGINS,
    AudioPlugin,
    BvhPlugin,
    CharacterPlugin,
    FogPlugin,
    LinesPlugin,
    StandardPhysicsPlugin,
    OrbitPlugin,
    OutlinePlugin,
    PlayerPlugin,
    ProfilePlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
    VignettePlugin,
];

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await withTimeout("WebGPU global setup", setupGlobals(), 5000);
const createCanvasContext = CanvasContext as unknown as new (
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
) => GPUCanvasContext;
if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
        observe() {}
        disconnect() {}
    } as unknown as typeof ResizeObserver;
}

function withTimeout<T>(label: string, promise: PromiseLike<T> | T, timeoutMs: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`${label} timed out after ${timeoutMs} ms`)),
            timeoutMs,
        );
        Promise.resolve(promise).then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error: unknown) => {
                clearTimeout(timer);
                reject(new Error(`${label} rejected: ${String(error)}`, { cause: error }));
            },
        );
    });
}

function watchDevice(device: GPUDevice) {
    let firstError: string | null = null;
    const waiters = new Set<(message: string) => void>();
    const uncaptured = (event: Event) => {
        event.preventDefault();
        const error = (event as GPUUncapturedErrorEvent).error;
        const kind = error.constructor.name || "GPUError";
        firstError ??= `${kind}: ${error.message}`;
        for (const reject of waiters) reject(firstError);
    };
    device.addEventListener("uncapturederror", uncaptured);
    return {
        wait<T>(label: string, promise: PromiseLike<T> | T, timeoutMs = 5000): Promise<T> {
            if (firstError)
                return Promise.reject(new Error(`${label}: uncaptured GPU error: ${firstError}`));
            return new Promise<T>((resolve, reject) => {
                const cleanup = () => {
                    clearTimeout(timer);
                    waiters.delete(onError);
                };
                const onError = (message: string) => {
                    cleanup();
                    reject(new Error(`${label}: uncaptured GPU error: ${message}`));
                };
                const timer = setTimeout(() => {
                    cleanup();
                    reject(new Error(`${label} timed out after ${timeoutMs} ms`));
                }, timeoutMs);
                waiters.add(onError);
                Promise.resolve(promise).then(
                    (value) => {
                        cleanup();
                        if (firstError)
                            reject(new Error(`${label}: uncaptured GPU error: ${firstError}`));
                        else resolve(value);
                    },
                    (error: unknown) => {
                        cleanup();
                        reject(new Error(`${label} rejected: ${String(error)}`, { cause: error }));
                    },
                );
            });
        },
        check(label: string): void {
            if (firstError) throw new Error(`${label}: uncaptured GPU error: ${firstError}`);
        },
        dispose(): void {
            device.removeEventListener("uncapturederror", uncaptured);
        },
    };
}

async function trackedDevice() {
    const adapter = await withTimeout("Dawn adapter request", navigator.gpu.requestAdapter(), 5000);
    if (!adapter) throw new Error("Dawn adapter unavailable");
    const requiredLimits: Record<string, number> = { maxStorageBuffersPerShaderStage: 10 };
    for (const limit of [
        "maxStorageBuffersInVertexStage",
        "maxStorageBuffersInFragmentStage",
        "maxStorageTexturesInVertexStage",
        "maxStorageTexturesInFragmentStage",
    ] as const) {
        if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
    }
    const device = await withTimeout(
        "Dawn device request",
        adapter.requestDevice({
            requiredFeatures: ["indirect-first-instance", "rg11b10ufloat-renderable"],
            requiredLimits,
        }),
        5000,
    );
    const watch = watchDevice(device);
    const live = new Set<GPUBuffer | GPUTexture>();
    const origins = new WeakMap<object, { owner?: World; label: string; createdAt: string }>();
    const labels = new WeakMap<World, string>();
    let checkingOwners = true;
    let buildScope = false;
    let creationOwnerOverride: World | undefined;
    let hasCreationOwnerOverride = false;
    const ownerAtCreation = (): World | undefined =>
        hasCreationOwnerOverride ? creationOwnerOverride : undefined;
    const resourceLabel = (resource: object | undefined): string => {
        if (!resource) return "missing GPU resource";
        return (
            (resource as { label?: string }).label || resource.constructor?.name || "GPU resource"
        );
    };
    const recordOrigin = <T extends object>(
        resource: T,
        owner: World | undefined,
        label?: string,
    ): T => {
        if (checkingOwners)
            origins.set(resource, {
                owner,
                label: label || resourceLabel(resource),
                createdAt: owner ? "inside a World callback" : "outside a World callback",
            });
        return resource;
    };
    const track = <T extends object>(resource: T, label?: string): T =>
        recordOrigin(resource, ownerAtCreation(), label);
    const assertOwned = (operation: string, resource: object | undefined): void => {
        if (!checkingOwners) return;
        const active = creationOwnerOverride;
        if (!active) return;
        const origin = resource ? origins.get(resource) : undefined;
        const activeName = labels.get(active) ?? "the active World";
        if (!origin)
            throw new Error(
                `${operation}: ${activeName} references untracked resource "${resourceLabel(resource)}" (creation location unknown)`,
            );
        if (!origin.owner)
            throw new Error(
                `${operation}: ${activeName} references ownerless resource "${origin.label}" created ${origin.createdAt}`,
            );
        if (origin.owner !== active) {
            const ownerName = labels.get(origin.owner) ?? "another World";
            throw new Error(
                `${operation}: ${activeName} references resource "${origin.label}" created by ${ownerName} (${origin.createdAt})`,
            );
        }
    };
    const bindGroupResource = (resource: GPUBindingResource): object | undefined => {
        if (typeof resource !== "object" || resource === null) return undefined;
        return "buffer" in resource ? (resource as GPUBufferBinding).buffer : resource;
    };
    const createBuffer = device.createBuffer.bind(device);
    const createTexture = device.createTexture.bind(device);
    const createSampler = device.createSampler.bind(device);
    const createBindGroup = device.createBindGroup.bind(device);
    const createCommandEncoder = device.createCommandEncoder.bind(device);
    Object.defineProperties(device, {
        createBuffer: {
            configurable: true,
            writable: true,
            value: (descriptor: GPUBufferDescriptor) => {
                const buffer = createBuffer(descriptor);
                live.add(buffer);
                track(buffer, descriptor.label);
                const destroy = buffer.destroy.bind(buffer);
                buffer.destroy = () => {
                    if (live.delete(buffer)) destroy();
                };
                return buffer;
            },
        },
        createTexture: {
            configurable: true,
            writable: true,
            value: (descriptor: GPUTextureDescriptor) => {
                const texture = createTexture(descriptor);
                live.add(texture);
                const owner = ownerAtCreation();
                recordOrigin(texture, owner, descriptor.label);
                const createView = texture.createView.bind(texture);
                texture.createView = (viewDescriptor?: GPUTextureViewDescriptor) =>
                    recordOrigin(
                        createView(viewDescriptor),
                        owner,
                        viewDescriptor?.label ||
                            (descriptor.label ? `${descriptor.label} view` : undefined),
                    );
                const destroy = texture.destroy.bind(texture);
                texture.destroy = () => {
                    if (live.delete(texture)) destroy();
                };
                return texture;
            },
        },
        createSampler: {
            configurable: true,
            writable: true,
            value: (descriptor?: GPUSamplerDescriptor) =>
                track(createSampler(descriptor), descriptor?.label),
        },
        createBindGroup: {
            configurable: true,
            writable: true,
            value: (descriptor: GPUBindGroupDescriptor) => {
                for (const entry of descriptor.entries)
                    assertOwned("createBindGroup entry", bindGroupResource(entry.resource));
                return createBindGroup(descriptor);
            },
        },
        createCommandEncoder: {
            configurable: true,
            writable: true,
            value: (descriptor?: GPUCommandEncoderDescriptor) => {
                const encoder = createCommandEncoder(descriptor);
                const copyBufferToBuffer = encoder.copyBufferToBuffer.bind(encoder);
                const copyBufferToTexture = encoder.copyBufferToTexture.bind(encoder);
                const copyTextureToBuffer = encoder.copyTextureToBuffer.bind(encoder);
                const copyTextureToTexture = encoder.copyTextureToTexture.bind(encoder);
                Object.defineProperties(encoder, {
                    copyBufferToBuffer: {
                        configurable: true,
                        writable: true,
                        value: (...args: Parameters<GPUCommandEncoder["copyBufferToBuffer"]>) => {
                            assertOwned("copyBufferToBuffer source", args[0]);
                            assertOwned("copyBufferToBuffer destination", args[2]);
                            return copyBufferToBuffer(...args);
                        },
                    },
                    copyBufferToTexture: {
                        configurable: true,
                        writable: true,
                        value: (...args: Parameters<GPUCommandEncoder["copyBufferToTexture"]>) => {
                            assertOwned("copyBufferToTexture source", args[0].buffer);
                            assertOwned("copyBufferToTexture destination", args[1].texture);
                            return copyBufferToTexture(...args);
                        },
                    },
                    copyTextureToBuffer: {
                        configurable: true,
                        writable: true,
                        value: (...args: Parameters<GPUCommandEncoder["copyTextureToBuffer"]>) => {
                            assertOwned("copyTextureToBuffer source", args[0].texture);
                            assertOwned("copyTextureToBuffer destination", args[1].buffer);
                            return copyTextureToBuffer(...args);
                        },
                    },
                    copyTextureToTexture: {
                        configurable: true,
                        writable: true,
                        value: (...args: Parameters<GPUCommandEncoder["copyTextureToTexture"]>) => {
                            assertOwned("copyTextureToTexture source", args[0].texture);
                            assertOwned("copyTextureToTexture destination", args[1].texture);
                            return copyTextureToTexture(...args);
                        },
                    },
                });
                return encoder;
            },
        },
    });
    const queue = device.queue;
    const writeBuffer = queue.writeBuffer.bind(queue);
    const writeTexture = queue.writeTexture.bind(queue);
    Object.defineProperties(queue, {
        writeBuffer: {
            configurable: true,
            writable: true,
            value: (...args: Parameters<GPUQueue["writeBuffer"]>) => {
                assertOwned("writeBuffer destination", args[0]);
                return writeBuffer(...args);
            },
        },
        writeTexture: {
            configurable: true,
            writable: true,
            value: (...args: Parameters<GPUQueue["writeTexture"]>) => {
                assertOwned("writeTexture destination", args[0].texture);
                return writeTexture(...args);
            },
        },
    });
    return {
        device,
        live,
        watch,
        labels,
        observeBuild(world: World): void {
            creationOwnerOverride = world;
            hasCreationOwnerOverride = true;
        },
        async withBuild<T>(callback: () => Promise<T>): Promise<T> {
            const previousBuildScope = buildScope;
            const previousOverride = creationOwnerOverride;
            const previousOverrideSet = hasCreationOwnerOverride;
            buildScope = true;
            creationOwnerOverride = undefined;
            hasCreationOwnerOverride = false;
            try {
                return await callback();
            } finally {
                buildScope = previousBuildScope;
                creationOwnerOverride = previousOverride;
                hasCreationOwnerOverride = previousOverrideSet;
            }
        },
        withWorld<T>(world: World, callback: () => T): T {
            const previousBuildScope = buildScope;
            const previousOverride = creationOwnerOverride;
            const previousOverrideSet = hasCreationOwnerOverride;
            buildScope = false;
            creationOwnerOverride = world;
            hasCreationOwnerOverride = true;
            try {
                return callback();
            } finally {
                buildScope = previousBuildScope;
                creationOwnerOverride = previousOverride;
                hasCreationOwnerOverride = previousOverrideSet;
            }
        },
        async withoutOwnershipChecks<T>(callback: () => Promise<T>): Promise<T> {
            const previous = checkingOwners;
            checkingOwners = false;
            try {
                return await callback();
            } finally {
                checkingOwners = previous;
            }
        },
    };
}

function addBody(world: World, y: number): number {
    const eid = world.create();
    world.add(eid, Body);
    const body = world.storage(Body);
    body.shape.set(eid, ShapeKind.Box);
    body.position.set(eid, 0, y, 0, 0);
    body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    body.type.set(eid, BodyType.Dynamic);
    return eid;
}

function addSpring(world: World, a: number, b: number): void {
    const eid = world.create();
    world.add(eid, DistanceJoint);
    const spring = world.storage(DistanceJoint);
    spring.a.set(eid, a);
    spring.b.set(eid, b);
    spring.enableSpring.set(eid, 1);
    spring.hertz.set(eid, 1);
    spring.dampingRatio.set(eid, 1);
    spring.length.set(eid, 1);
}

function addJoint(world: World, a: number, b: number): void {
    const eid = world.create();
    world.add(eid, SphericalJoint);
    const joint = world.storage(SphericalJoint);
    joint.a.set(eid, a);
    joint.b.set(eid, b);
}

function expectStateViews(world: World, eids: number[]): void {
    expect(eids.length).toBeGreaterThan(0);
    expect(eids.every((eid) => world.resource(Views).get(eid) !== undefined)).toBe(true);
}

async function stepGpuWorld(
    world: World,
    label: string,
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
): Promise<void> {
    tracked.withWorld(world, () => world.step(Time.FIXED_DT));
    await tracked.watch.wait(
        `${label} frame submission`,
        tracked.device.queue.onSubmittedWorkDone(),
    );
    tracked.watch.check(`${label} frame submission`);
}

interface IsolationResources {
    camera: number;
    actor: number;
    part: number;
    sky: number;
    bvh: Bvh | null;
}

const ISOLATION_FONT = `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`;
const isolationKey = { create: () => createIsolationResources() };
const createIsolationResources = (): IsolationResources => ({
    camera: -1,
    actor: -1,
    part: -1,
    sky: -1,
    bvh: null,
});

function uses(subject: Plugin, dependency: Plugin): boolean {
    return (
        subject === dependency ||
        (subject.dependencies ?? []).some((plugin) => uses(plugin, dependency))
    );
}

function featurePlugin(subject: Plugin): Plugin {
    const components = everyPlugin.flatMap((plugin) => plugin.components ?? []);
    return {
        name: "GpuIsolationFeatureSeed",
        gpu: {},
        components,
        // Character's app composes Physics explicitly; this fixture is that app.
        dependencies: [
            ...DEFAULT_PLUGINS,
            ...(uses(subject, CharacterPlugin) ? [StandardPhysicsPlugin] : []),
            subject,
        ],
        initialize(world) {
            const resources = world.resource(isolationKey);
            let context: GPUCanvasContext;
            const canvas = {
                width: 32,
                height: 24,
                style: { imageRendering: "auto" },
                getContext(kind: string) {
                    return kind === "webgpu" ? context : null;
                },
                getBoundingClientRect() {
                    return { width: 32, height: 24 };
                },
            } as unknown as HTMLCanvasElement;
            context = new createCanvasContext(canvas, canvas.width, canvas.height);

            const camera = world.create();
            resources.camera = camera;
            world.add(camera, Transform);
            world.add(camera, Camera);
            world.add(camera, Resolution);
            world.add(camera, StandardRenderer);
            world.add(camera, DepthPrepass);
            world.add(camera, CameraBackground);
            world.add(camera, Tonemapping);
            world.add(camera, Vignette, { intensity: 0.1 });
            world.add(camera, Orbit);
            world.add(camera, Listener);
            world.storage(Transform).translation.set(camera, 0, 4, 12, 0);
            world
                .storage(CameraBackground)
                .name.set(camera, world.resource(Backgrounds).id("sky") ?? 0);
            attachCanvas(camera, canvas, world);

            const ambient = world.create();
            world.add(ambient, AmbientLight);
            if (
                subject === RenderingPlugin ||
                subject === StandardRenderingPlugin ||
                subject.name === "CorePipeline" ||
                uses(subject, SkyPlugin)
            ) {
                const sun = world.create();
                world.add(sun, DirectionalLight);
                world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
                world.add(sun, VolumetricLight);
                const point = world.create();
                world.add(point, Transform);
                world.add(point, SpotLight, { shadowMapsEnabled: 1 });
                world.add(point, VolumetricLight);
                world.storage(Transform).translation.set(point, 1, 2, 1, 0);
            }

            if (subject === RenderingPlugin || subject === StandardRenderingPlugin) {
                expect([...world.query([DirectionalLight, VolumetricLight])]).toHaveLength(1);
                expect([...world.query([SpotLight, VolumetricLight])]).toHaveLength(1);
            }

            const sky = world.create();
            resources.sky = sky;
            world.add(sky, Sky);
            const fog = world.create();
            world.add(fog, Fog);

            const part = world.create();
            resources.part = part;
            world.add(part, Transform);
            world.add(part, MeshInstance);
            world.add(part, MeshMaterial, {
                material: world.resource(Materials).add(
                    StandardMaterial({
                        baseColor: [0.8, 0.25, 0.1, 1],
                        metallic: 0.1,
                        perceptualRoughness: 0.6,
                    }),
                ),
            });
            world.add(part, Outline);
            world.storage(Transform).translation.set(part, 0, 1, 0, 0);

            const line = world.create();
            world.add(line, Transform);
            world.add(line, Line);
            world.add(line, Arrow);
            world.storage(Transform).translation.set(line, -1, 0, 0, 0);

            const sprite = world.create();
            world.add(sprite, Transform);
            world.add(sprite, Sprite);
            world.storage(Transform).translation.set(sprite, 1, 0, 0, 0);

            const label = world.create();
            world.add(label, Transform);
            world.add(label, Text);
            world.storage(Transform).translation.set(label, 0, 2, 0, 0);
            if (uses(subject, TextPlugin))
                world
                    .storage(Text)
                    .font.set(label, registerFont(world, ISOLATION_FONT, "isolation"));
            world.storage(Text).content.set(label, internText(world, "isolation"));

            const sound = world.create();
            world.add(sound, Sound);

            const actor = world.create();
            resources.actor = actor;
            world.add(actor, Body);
            world.add(actor, Character);
            world.add(actor, Player);
            world.storage(Body).shape.set(actor, ShapeKind.Capsule);
            world.storage(Body).position.set(actor, 0, 2, 2, 0);
            world.storage(Body).halfExtents.set(actor, 0, 0.6, 0, 0.35);
            world.storage(Body).type.set(actor, BodyType.Kinematic);
            world.storage(Player).camera.set(actor, camera);

            const globalTransforms = world.gpu.buffers.get("global-transform-interpolated");
            if (!globalTransforms)
                throw new Error("Engine GlobalTransform did not publish its renderer buffer");
        },
        async warm(world) {
            const resources = world.resource(isolationKey);
            if (!uses(subject, BvhPlugin)) return;
            const device = world.gpu.device;
            const bvh = await createBvh(world, device, 2);
            device.queue.writeBuffer(
                bvh.prims,
                0,
                new Float32Array([0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 3, 1, 1, 0]),
            );
            device.queue.writeBuffer(bvh.count, 0, new Uint32Array([2]));
            const encoder = device.createCommandEncoder({ label: "gpu-isolation-bvh" });
            bvh.build(encoder);
            device.queue.submit([encoder.finish()]);
            resources.bvh = bvh;
        },
        dispose(world) {
            const resources = world.resource(isolationKey);
            resources.bvh?.destroy();
            resources.bvh = null;
        },
    };
}

type SceneColor = readonly [number, number, number, number];
interface IsolationContent {
    actorY: number;
    clearColor: number;
    skyZenith: number;
    skyHorizon: number;
    bodyHeights: readonly [number, number];
    color: SceneColor;
}
const FIRST_CONTENT: IsolationContent = {
    actorY: 4,
    clearColor: 0xc04020,
    skyZenith: 0xc04020,
    skyHorizon: 0xf08020,
    bodyHeights: [2, 3],
    color: [0.8, 0.2, 0.1, 1],
};
const SECOND_CONTENT: IsolationContent = {
    actorY: 8,
    clearColor: 0x2040c0,
    skyZenith: 0x2040c0,
    skyHorizon: 0x40c0f0,
    bodyHeights: [20, 21],
    color: [0.1, 0.25, 0.8, 1],
};

function authorIsolationContent(
    world: World,
    resources: IsolationResources,
    content: IsolationContent,
): number {
    const a = addBody(world, content.bodyHeights[0]);
    const b = addBody(world, content.bodyHeights[1]);
    addSpring(world, a, b);
    addJoint(world, a, b);
    (() => {
        world.storage(Body).position.y.set(resources.actor, content.actorY);
        world.storage(Camera).clearColor.set(resources.camera, content.clearColor);
        world.storage(Sky).zenith.set(resources.sky, content.skyZenith);
        world.storage(Sky).horizon.set(resources.sky, content.skyHorizon);
        const materials = world.resource(Materials);
        const material = world.storage(MeshMaterial).material.get(resources.part);
        materials.update(material, { baseColor: content.color });
    })();
    return a;
}

async function readRenderedFrame(
    world: World,
    resources: IsolationResources,
    label: string,
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
): Promise<Uint8Array> {
    const texture = offscreenTexture(world, resources.camera);
    if (!texture) throw new Error(`${label}: camera has no World-owned offscreen texture`);
    const probe = await tracked.withoutOwnershipChecks(() =>
        tracked.watch.wait(
            `${label} probeTexture readback`,
            probeTexture(world, texture, { label }),
        ),
    );
    return new Uint8Array(probe.bytes);
}

async function renderAlone(
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
    content: IsolationContent,
    label: string,
    subject: Plugin,
): Promise<Uint8Array> {
    const app = await tracked.withBuild(() =>
        tracked.watch.wait(
            `${label} solo world build`,
            createApp({
                defaults: false,
                plugins: [featurePlugin(subject)],
                device: tracked.device,
                setup: (world) => tracked.observeBuild(world),
            }),
        ),
    );
    try {
        tracked.labels.set(app.world, label);
        const resources = app.world.resource(isolationKey);
        tracked.withWorld(app.world, () => authorIsolationContent(app.world, resources, content));
        await stepGpuWorld(app.world, `${label} solo world`, tracked);
        return await readRenderedFrame(app.world, resources, `${label} solo frame`, tracked);
    } finally {
        app.dispose();
    }
}

async function exerciseIsolationPair(sharedDevice: boolean, subject: Plugin): Promise<void> {
    const hasPhysics = uses(subject, StandardPhysicsPlugin) || uses(subject, CharacterPlugin);
    const firstDevice = await trackedDevice();
    const secondDevice = sharedDevice ? firstDevice : await trackedDevice();
    const seed = featurePlugin(subject);
    let first: Awaited<ReturnType<typeof createApp>> | undefined;
    let second: Awaited<ReturnType<typeof createApp>> | undefined;
    let firstPairPixels: Uint8Array | undefined;
    let secondPairPixels: Uint8Array | undefined;
    let prebuildBuffer: GPUBuffer | undefined;
    try {
        prebuildBuffer = secondDevice.device.createBuffer({
            label: "reviewer pre-build buffer",
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
        });
        first = await firstDevice.withBuild(() =>
            firstDevice.watch.wait(
                "first world build",
                createApp({
                    defaults: false,
                    plugins: [seed],
                    device: firstDevice.device,
                    setup: (world) => firstDevice.observeBuild(world),
                }),
            ),
        );
        firstDevice.labels.set(first.world, "first world");
        const firstFeatures = first.world.resource(isolationKey);
        const firstA = firstDevice.withWorld(first.world, () =>
            authorIsolationContent(first!.world, firstFeatures, FIRST_CONTENT),
        );

        second = await secondDevice.withBuild(() =>
            secondDevice.watch.wait(
                "second world build",
                createApp({
                    defaults: false,
                    plugins: [seed],
                    device: secondDevice.device,
                    setup: (world) => secondDevice.observeBuild(world),
                }),
            ),
        );
        secondDevice.labels.set(second.world, "second world");
        const secondFeatures = second.world.resource(isolationKey);
        let prebuildWriteError: unknown;
        try {
            secondDevice.withWorld(second.world, () =>
                secondDevice.device.queue.writeBuffer(prebuildBuffer!, 0, new Uint8Array(16)),
            );
        } catch (error) {
            prebuildWriteError = error;
        }
        expect(String(prebuildWriteError)).toContain("reviewer pre-build buffer");
        expect(String(prebuildWriteError)).toContain("created outside a World callback");
        const peerA = secondDevice.withWorld(second.world, () =>
            authorIsolationContent(second!.world, secondFeatures, SECOND_CONTENT),
        );

        await stepGpuWorld(first.world, "first world", firstDevice);
        await stepGpuWorld(second.world, "second world", secondDevice);
        firstDevice.watch.check("first world GPU work");
        secondDevice.watch.check("second world GPU work");
        const firstTexture = offscreenTexture(first.world, firstFeatures.camera);
        const secondTexture = offscreenTexture(second.world, secondFeatures.camera);
        if (!firstTexture || !secondTexture)
            throw new Error("both Worlds must own their rendered offscreen texture");
        expect(firstTexture).not.toBe(secondTexture);
        firstPairPixels = await readRenderedFrame(
            first.world,
            firstFeatures,
            "first world frame",
            firstDevice,
        );
        secondPairPixels = await readRenderedFrame(
            second.world,
            secondFeatures,
            "second world frame",
            secondDevice,
        );
        expect(firstPairPixels).not.toEqual(secondPairPixels);
        if (hasPhysics) {
            expect(physicsWorld(first.world)?.getCounters().jointCount).toBe(2);
            expect(physicsWorld(second.world)?.getCounters().jointCount).toBe(2);
            const firstHash = hashPhysics(first.world);
            const firstBody = readBody(first.world, firstA);
            if (!firstBody) throw new Error("first Physics body did not become live");
            const siblingHash = hashPhysics(second.world);
            const siblingBody = readBody(second.world, peerA);
            const saved = first.world.snapshot();
            setVelocity(first.world, firstA, 7, 0, 0);
            expect(readBody(first.world, firstA)?.linearVelocity[0]).toBeCloseTo(7);
            first.world.restore(saved);
            expect(hashPhysics(first.world)).toBe(firstHash);
            expect(readBody(first.world, firstA)).toEqual(firstBody);
            expect(hashPhysics(second.world)).toBe(siblingHash);
            expect(readBody(second.world, peerA)).toEqual(siblingBody);

            if (uses(subject, CharacterPlugin)) {
                expect(first.world.has(firstFeatures.actor, GlobalTransform)).toBe(true);
                expect(second.world.has(secondFeatures.actor, GlobalTransform)).toBe(true);
                expect(
                    first.world.storage(GlobalTransform).translation.y.get(firstFeatures.actor),
                ).not.toBe(
                    second.world.storage(GlobalTransform).translation.y.get(secondFeatures.actor),
                );
            }
        }
        if (uses(subject, BvhPlugin)) {
            expect(firstFeatures.bvh).not.toBeNull();
            expect(secondFeatures.bvh).not.toBeNull();
        }
        if (uses(subject, SkyPlugin)) {
            expectStateViews(first.world, cascadeComboEids(first.world));
            expectStateViews(second.world, cascadeComboEids(second.world));
            expectStateViews(first.world, pointComboEids(first.world));
            expectStateViews(second.world, pointComboEids(second.world));
        }
        if (hasPhysics) {
            expect([...first.world.query([GlobalTransform])].length).toBeGreaterThan(0);
            expect([...second.world.query([GlobalTransform])].length).toBeGreaterThan(0);
        }
        for (const [plugin, key] of [
            [SpritePlugin, "spriteData"],
            [TextPlugin, "textGlyphs"],
            [LinesPlugin, "lineSegments"],
            [SkyPlugin, "sky"],
        ] as const) {
            if (!uses(subject, plugin)) continue;
            const a = first.world.gpu.buffers.get(key);
            const b = second.world.gpu.buffers.get(key);
            expect(a).toBeDefined();
            expect(b).toBeDefined();
            expect(a).not.toBe(b);
        }

        const peerHashBeforeDispose = hasPhysics ? hashPhysics(second.world) : 0n;
        const peerBodyBeforeDispose = hasPhysics ? readBody(second.world, peerA) : null;
        const peerResources = new Set<GPUBuffer | GPUTexture>([
            ...second.world.gpu.buffers.values(),
            ...second.world.gpu.textures.values(),
        ]);
        const peerRegistries = {
            buffers: [...second.world.gpu.buffers],
            textures: [...second.world.gpu.textures],
            typed: [...second.world.gpu.typed],
        };

        first.dispose();
        first = undefined;
        expect([...peerResources].every((resource) => secondDevice.live.has(resource))).toBe(true);
        expect([...second.world.gpu.buffers]).toEqual(peerRegistries.buffers);
        expect([...second.world.gpu.textures]).toEqual(peerRegistries.textures);
        expect([...second.world.gpu.typed]).toEqual(peerRegistries.typed);
        if (hasPhysics) {
            expect(hashPhysics(second.world)).toBe(peerHashBeforeDispose);
            expect(readBody(second.world, peerA)).toEqual(peerBodyBeforeDispose);
        }
        await stepGpuWorld(second.world, "second world after sibling disposal", secondDevice);
        if (hasPhysics) expect(readBody(second.world, peerA)).not.toEqual(peerBodyBeforeDispose);

        second.dispose();
        second = undefined;
        if (!firstPairPixels || !secondPairPixels)
            throw new Error("both paired Worlds must produce a captured frame");
        const firstSoloPixels =
            subject === TextPlugin
                ? textBaselines.get(FIRST_CONTENT)!
                : await renderAlone(firstDevice, FIRST_CONTENT, "first world", subject);
        const secondSoloPixels =
            subject === TextPlugin
                ? textBaselines.get(SECOND_CONTENT)!
                : await renderAlone(secondDevice, SECOND_CONTENT, "second world", subject);
        expect(firstPairPixels).toEqual(firstSoloPixels);
        expect(secondPairPixels).toEqual(secondSoloPixels);
    } finally {
        try {
            second?.dispose();
            first?.dispose();
            prebuildBuffer?.destroy();
        } finally {
            firstDevice.watch.dispose();
            firstDevice.device.destroy();
            if (!sharedDevice) {
                secondDevice.watch.dispose();
                secondDevice.device.destroy();
            }
        }
    }
    expect(firstDevice.live.size).toBe(0);
    if (!sharedDevice) expect(secondDevice.live.size).toBe(0);
}

// Prepare independent Text witnesses; only owned CPU pixels survive, never a device or plugin GPU state.
const textBaselines = new Map<IsolationContent, Uint8Array>();
for (const content of [FIRST_CONTENT, SECOND_CONTENT]) {
    beforeAll(async () => {
        const tracked = await trackedDevice();
        try {
            textBaselines.set(
                content,
                await renderAlone(tracked, content, "Text solo witness", TextPlugin),
            );
        } finally {
            tracked.watch.dispose();
            tracked.device.destroy();
        }
        expect(tracked.live.size).toBe(0);
    });
}

for (const plugin of everyPlugin) {
    for (const shared of [true, false]) {
        test(`${plugin.name}: components and GPU paths stay isolated on ${shared ? "a shared device" : "separate devices"}`, async () => {
            await exerciseIsolationPair(shared, plugin);
        });
    }
}
