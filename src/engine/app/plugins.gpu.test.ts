import { expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    AmbientLight,
    attachCanvas,
    Backgrounds,
    Camera,
    DirectionalLight,
    PointLight,
    Resolution,
    Spot,
    Views,
    Volumetric,
} from "../../core/rendering";
import { offscreenTexture } from "../../core/rendering/view";
import {
    Arrow,
    CellsPlugin,
    cellsGridFor,
    Fog,
    FogPlugin,
    GltfPlugin,
    Line,
    LinesPlugin,
    liveSkin,
    loadGltf,
    Orbit,
    OrbitOverlayPlugin,
    OrbitPlugin,
    Outline,
    OutlinePlugin,
    PhysicsProfilePlugin,
    Player,
    PlayerPlugin,
    ProfilePlugin,
    placeScene,
    Skin,
    SkinPlugin,
    Sky,
    SkyPlugin,
    Sprite,
    SpritePlugin,
    Text,
    TextPlugin,
    text,
} from "../../extras";
import {
    Color,
    DEFAULT_PLUGINS,
    Glaze,
    Part,
    PartPlugin,
    Transform,
    TransformsPlugin,
} from "../../standard";
import {
    Backdrop,
    cascadeComboEids,
    Depth,
    Material,
    pointComboEids,
    Sear,
    Shadow,
    Tag,
} from "../../standard/rendering";
import { AudioPlugin, Listener, Sound } from "../../transitional/audio";
import { type Bvh, BvhPlugin, createBvh } from "../../transitional/bvh";
import { Character, CharacterPlugin, pose } from "../../transitional/character";
import { type Mirror, MirrorPlugin, mirror } from "../../transitional/mirror";
import {
    Body,
    hash as hashPhysics,
    Joint,
    PhysicsPlugin,
    Pose,
    physicsWorld,
    readBody,
    restore as restorePhysics,
    ShapeKind,
    Spring,
    setVelocity,
    snapshot as snapshotPhysics,
} from "../../transitional/physics";
import { Compute, type Plugin, probeTexture, type State, Time } from "../index";
import { currentWorld, withCompute } from "../runtime";
import { build } from "./index";

const everyPlugin: readonly Plugin[] = [
    ...DEFAULT_PLUGINS,
    AudioPlugin,
    BvhPlugin,
    CharacterPlugin,
    CellsPlugin,
    FogPlugin,
    GltfPlugin,
    LinesPlugin,
    MirrorPlugin,
    PhysicsPlugin,
    OrbitOverlayPlugin,
    OrbitPlugin,
    OutlinePlugin,
    PhysicsProfilePlugin,
    PlayerPlugin,
    ProfilePlugin,
    SkinPlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
];

const peerModule = "bun-webgpu";
const peer = (await import(peerModule)) as Record<string, unknown> & {
    setupGlobals(): Promise<void>;
};
await withTimeout("WebGPU global setup", peer.setupGlobals(), 4_000);
const createCanvasContext = peer.GPUCanvasContextMock as new (
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
        wait<T>(label: string, promise: PromiseLike<T> | T, timeoutMs = 4_000): Promise<T> {
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
    const adapter = await withTimeout(
        "Dawn adapter request",
        navigator.gpu.requestAdapter(),
        4_000,
    );
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
            requiredFeatures: ["bgra8unorm-storage", "rg11b10ufloat-renderable", "timestamp-query"],
            requiredLimits,
        }),
        4_000,
    );
    const watch = watchDevice(device);
    const live = new Set<GPUBuffer | GPUTexture>();
    const origins = new WeakMap<object, { owner?: State; label: string; createdAt: string }>();
    const labels = new WeakMap<State, string>();
    let checkingOwners = true;
    let buildScope = false;
    let creationOwnerOverride: State | undefined;
    let hasCreationOwnerOverride = false;
    const ownerAtCreation = (): State | undefined =>
        hasCreationOwnerOverride
            ? creationOwnerOverride
            : buildScope
              ? currentWorld<State>()
              : undefined;
    const resourceLabel = (resource: object | undefined): string => {
        if (!resource) return "missing GPU resource";
        return (
            (resource as { label?: string }).label || resource.constructor?.name || "GPU resource"
        );
    };
    const recordOrigin = <T extends object>(
        resource: T,
        owner: State | undefined,
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
        const active = currentWorld<State>();
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
        withWorld<T>(state: State, callback: () => T): T {
            const previousBuildScope = buildScope;
            const previousOverride = creationOwnerOverride;
            const previousOverrideSet = hasCreationOwnerOverride;
            buildScope = false;
            creationOwnerOverride = state;
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

function addBody(state: State, y: number): number {
    const eid = state.create();
    state.add(eid, Body);
    const body = state.of(Body);
    body.shape.set(eid, ShapeKind.Box);
    body.pos.set(eid, 0, y, 0, 0);
    body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    body.mass.set(eid, 1);
    return eid;
}

function addSpring(state: State, a: number, b: number): void {
    const eid = state.create();
    state.add(eid, Spring);
    const spring = state.of(Spring);
    spring.a.set(eid, a);
    spring.b.set(eid, b);
    spring.stiffness.set(eid, 10);
    spring.rest.set(eid, 1);
}

function addJoint(state: State, a: number, b: number): void {
    const eid = state.create();
    state.add(eid, Joint);
    const joint = state.of(Joint);
    joint.a.set(eid, a);
    joint.b.set(eid, b);
}

function expectStateViews(state: State, eids: number[]): void {
    expect(eids.length).toBeGreaterThan(0);
    expect(withCompute(state.gpu, () => eids.every((eid) => Views.get(eid) !== undefined))).toBe(
        true,
    );
}

async function stepGpuWorld(
    state: State,
    label: string,
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
): Promise<void> {
    tracked.withWorld(state, () => state.step(Time.FIXED_DT));
    await tracked.watch.wait(
        `${label} frame submission`,
        tracked.device.queue.onSubmittedWorkDone(),
    );
    tracked.watch.check(`${label} frame submission`);
}

async function waitForMirrorMap(mirror: Mirror): Promise<void> {
    const deadline = Date.now() + 1_000;
    while (!mirror.snapshot) {
        if (Date.now() >= deadline)
            throw new Error("Mirror mapAsync did not produce a snapshot in 1000 ms");
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
}

interface IsolationResources {
    camera: number;
    actor: number;
    part: number;
    sky: number;
    mirror: Mirror | null;
    bvh: Bvh | null;
    gltfInstances: number[];
}

const isolationKey = Symbol("gpu-isolation");
const createIsolationResources = (): IsolationResources => ({
    camera: -1,
    actor: -1,
    part: -1,
    sky: -1,
    mirror: null,
    bvh: null,
    gltfInstances: [],
});

function featurePlugin(): Plugin {
    return {
        name: "GpuIsolationFeatureSeed",
        dependencies: [
            PartPlugin,
            SkinPlugin,
            MirrorPlugin,
            TransformsPlugin,
            GltfPlugin,
            SkyPlugin,
        ],
        initialize(state) {
            const resources = state.resource(isolationKey, createIsolationResources);
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

            const camera = state.create();
            resources.camera = camera;
            state.add(camera, Transform);
            state.add(camera, Camera);
            state.add(camera, Resolution);
            state.add(camera, Sear);
            state.add(camera, Tag);
            state.add(camera, Depth);
            state.add(camera, Backdrop);
            state.add(camera, Glaze);
            state.add(camera, Orbit);
            state.add(camera, Listener);
            Transform.pos.set(camera, 0, 4, 12, 0);
            Backdrop.name.set(camera, Backgrounds.id("sky") ?? 0);
            attachCanvas(camera, canvas, state);

            const ambient = state.create();
            state.add(ambient, AmbientLight);
            const sun = state.create();
            state.add(sun, DirectionalLight);
            state.add(sun, Shadow);
            state.add(sun, Volumetric);
            const point = state.create();
            state.add(point, Transform);
            state.add(point, PointLight);
            state.add(point, Spot);
            state.add(point, Shadow);
            state.add(point, Volumetric);
            Transform.pos.set(point, 1, 2, 1, 0);

            const sky = state.create();
            resources.sky = sky;
            state.add(sky, Sky);
            const fog = state.create();
            state.add(fog, Fog);

            const part = state.create();
            resources.part = part;
            state.add(part, Transform);
            state.add(part, Part);
            state.add(part, Color);
            state.add(part, Material);
            state.add(part, Outline);
            state.add(part, Skin);
            Transform.pos.set(part, 0, 1, 0, 0);
            Color.rgba.set(part, 0.8, 0.25, 0.1, 1);
            Material.params.set(part, 0.1, 0.6, 0, 1);
            const skin = liveSkin(state);
            Skin.anim.x.set(part, skin.alloc(part, 1, state.stamp(part)));
            skin.flush(Compute.device);

            const line = state.create();
            state.add(line, Transform);
            state.add(line, Line);
            state.add(line, Arrow);
            Transform.pos.set(line, -1, 0, 0, 0);

            const sprite = state.create();
            state.add(sprite, Transform);
            state.add(sprite, Sprite);
            Transform.pos.set(sprite, 1, 0, 0, 0);

            const label = state.create();
            state.add(label, Transform);
            state.add(label, Text);
            Transform.pos.set(label, 0, 2, 0, 0);
            Text.content.set(label, text("isolation"));

            const sound = state.create();
            state.add(sound, Sound);

            const actor = state.create();
            resources.actor = actor;
            state.add(actor, Body);
            state.add(actor, Character);
            state.add(actor, Player);
            Body.shape.set(actor, ShapeKind.Capsule);
            Body.pos.set(actor, 0, 2, 2, 0);
            Body.halfExtents.set(actor, 0, 0.6, 0, 0.35);
            Body.mass.set(actor, 0);
            Player.camera.set(actor, camera);

            const transforms = Compute.buffers.get("transforms");
            if (!transforms) throw new Error("Transforms did not publish their buffer");
            resources.mirror = mirror(state, transforms);
        },
        async warm(state) {
            const resources = state.resource(isolationKey, createIsolationResources);
            const imported = await loadGltf(
                state,
                resolve(import.meta.dir, "../../transitional/gltf/fixtures/box-meshopt.glb"),
            );
            resources.gltfInstances = placeScene(state, imported);
            const device = Compute.device;
            const bvh = await createBvh(device, 2);
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
        dispose(state) {
            const resources = state.resource(isolationKey, createIsolationResources);
            resources.mirror?.dispose();
            resources.mirror = null;
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
    state: State,
    resources: IsolationResources,
    content: IsolationContent,
): number {
    const a = addBody(state, content.bodyHeights[0]);
    const b = addBody(state, content.bodyHeights[1]);
    addSpring(state, a, b);
    addJoint(state, a, b);
    withCompute(state.gpu, () => {
        Body.pos.y.set(resources.actor, content.actorY);
        Camera.clearColor.set(resources.camera, content.clearColor);
        Sky.zenith.set(resources.sky, content.skyZenith);
        Sky.horizon.set(resources.sky, content.skyHorizon);
        Color.rgba.set(resources.part, ...content.color);
    });
    return a;
}

async function readRenderedFrame(
    state: State,
    resources: IsolationResources,
    label: string,
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
): Promise<Uint8Array> {
    const texture = offscreenTexture(state, resources.camera);
    if (!texture) throw new Error(`${label}: camera has no State-owned offscreen texture`);
    const probe = await tracked.withoutOwnershipChecks(() =>
        tracked.watch.wait(
            `${label} probeTexture readback`,
            probeTexture(tracked.device, texture, { label }),
        ),
    );
    return new Uint8Array(probe.bytes);
}

async function renderAlone(
    tracked: Awaited<ReturnType<typeof trackedDevice>>,
    content: IsolationContent,
    label: string,
): Promise<Uint8Array> {
    const app = await tracked.withBuild(() =>
        tracked.watch.wait(
            `${label} solo world build`,
            build({
                defaults: false,
                plugins: [...everyPlugin, featurePlugin()],
                device: tracked.device,
            }),
        ),
    );
    try {
        tracked.labels.set(app.state, label);
        const resources = app.state.resource(isolationKey, createIsolationResources);
        tracked.withWorld(app.state, () => authorIsolationContent(app.state, resources, content));
        await stepGpuWorld(app.state, `${label} solo world`, tracked);
        return await readRenderedFrame(app.state, resources, `${label} solo frame`, tracked);
    } finally {
        app.dispose();
    }
}

async function exerciseIsolationPair(sharedDevice: boolean): Promise<void> {
    const firstDevice = await trackedDevice();
    const secondDevice = sharedDevice ? firstDevice : await trackedDevice();
    const seed = featurePlugin();
    let first: Awaited<ReturnType<typeof build>> | undefined;
    let second: Awaited<ReturnType<typeof build>> | undefined;
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
                build({
                    defaults: false,
                    plugins: [...everyPlugin, seed],
                    device: firstDevice.device,
                }),
                4_000,
            ),
        );
        firstDevice.labels.set(first.state, "first world");
        const firstFeatures = first.state.resource(isolationKey, createIsolationResources);
        const firstA = firstDevice.withWorld(first.state, () =>
            authorIsolationContent(first!.state, firstFeatures, FIRST_CONTENT),
        );

        second = await secondDevice.withBuild(() =>
            secondDevice.watch.wait(
                "second world build",
                build({
                    defaults: false,
                    plugins: [...everyPlugin, seed],
                    device: secondDevice.device,
                }),
                4_000,
            ),
        );
        secondDevice.labels.set(second.state, "second world");
        const secondFeatures = second.state.resource(isolationKey, createIsolationResources);
        let prebuildWriteError: unknown;
        try {
            withCompute(second.state.gpu, () =>
                secondDevice.device.queue.writeBuffer(prebuildBuffer!, 0, new Uint8Array(16)),
            );
        } catch (error) {
            prebuildWriteError = error;
        }
        expect(String(prebuildWriteError)).toContain("reviewer pre-build buffer");
        expect(String(prebuildWriteError)).toContain("created outside a World callback");
        const peerA = secondDevice.withWorld(second.state, () =>
            authorIsolationContent(second!.state, secondFeatures, SECOND_CONTENT),
        );

        await stepGpuWorld(first.state, "first world", firstDevice);
        await stepGpuWorld(second.state, "second world", secondDevice);
        await firstDevice.watch.wait(
            "first-world Mirror mapAsync",
            waitForMirrorMap(firstFeatures.mirror!),
        );
        await secondDevice.watch.wait(
            "second-world Mirror mapAsync",
            waitForMirrorMap(secondFeatures.mirror!),
        );
        firstDevice.watch.check("first world GPU work");
        secondDevice.watch.check("second world GPU work");
        const firstTexture = offscreenTexture(first.state, firstFeatures.camera);
        const secondTexture = offscreenTexture(second.state, secondFeatures.camera);
        if (!firstTexture || !secondTexture)
            throw new Error("both Worlds must own their rendered offscreen texture");
        expect(firstTexture).not.toBe(secondTexture);
        firstPairPixels = await readRenderedFrame(
            first.state,
            firstFeatures,
            "first world frame",
            firstDevice,
        );
        secondPairPixels = await readRenderedFrame(
            second.state,
            secondFeatures,
            "second world frame",
            secondDevice,
        );
        expect(firstPairPixels).not.toEqual(secondPairPixels);
        expect(physicsWorld(first.state)?.getCounters().jointCount).toBe(2);
        expect(physicsWorld(second.state)?.getCounters().jointCount).toBe(2);
        const firstHash = hashPhysics(first.state);
        const firstBody = readBody(first.state, firstA);
        if (!firstBody) throw new Error("first Physics body did not become live");
        const siblingHash = hashPhysics(second.state);
        const siblingBody = readBody(second.state, peerA);
        const saved = snapshotPhysics(first.state);
        setVelocity(first.state, firstA, 7, 0, 0);
        expect(readBody(first.state, firstA)?.vel[0]).toBeCloseTo(7);
        restorePhysics(first.state, saved);
        expect(hashPhysics(first.state)).toBe(firstHash);
        expect(readBody(first.state, firstA)).toEqual(firstBody);
        expect(hashPhysics(second.state)).toBe(siblingHash);
        expect(readBody(second.state, peerA)).toEqual(siblingBody);

        const firstActorPose = [0, 0, 0] as [number, number, number];
        const secondActorPose = [0, 0, 0] as [number, number, number];
        expect(pose(first.state, firstFeatures.actor, firstActorPose)).toBe(true);
        expect(pose(second.state, secondFeatures.actor, secondActorPose)).toBe(true);
        expect(firstActorPose[1]).not.toBe(secondActorPose[1]);
        expect(firstFeatures.gltfInstances.length).toBeGreaterThan(0);
        expect(secondFeatures.gltfInstances.length).toBeGreaterThan(0);
        expect(firstFeatures.bvh).not.toBeNull();
        expect(secondFeatures.bvh).not.toBeNull();
        expect(firstFeatures.mirror?.allocated).toBeGreaterThan(0);
        expect(secondFeatures.mirror?.allocated).toBeGreaterThan(0);
        expectStateViews(first.state, cascadeComboEids(first.state));
        expectStateViews(second.state, cascadeComboEids(second.state));
        expectStateViews(first.state, pointComboEids(first.state));
        expectStateViews(second.state, pointComboEids(second.state));
        expect([...first.state.query([Pose])].length).toBeGreaterThan(0);
        expect([...second.state.query([Pose])].length).toBeGreaterThan(0);
        expect(
            withCompute(first.state.gpu, () => cellsGridFor(firstFeatures.camera)),
        ).toBeDefined();
        expect(
            withCompute(second.state.gpu, () => cellsGridFor(secondFeatures.camera)),
        ).toBeDefined();

        for (const key of ["skinData", "spriteData", "textGlyphs", "lineSegments", "sky"]) {
            const a = first.state.gpu.buffers.get(key);
            const b = second.state.gpu.buffers.get(key);
            expect(a).toBeDefined();
            expect(b).toBeDefined();
            expect(a).not.toBe(b);
        }

        const peerHashBeforeDispose = hashPhysics(second.state);
        const peerBodyBeforeDispose = readBody(second.state, peerA);
        const peerResources = new Set<GPUBuffer | GPUTexture>([
            ...second.state.gpu.buffers.values(),
            ...second.state.gpu.textures.values(),
        ]);
        const peerRegistries = {
            buffers: [...second.state.gpu.buffers],
            textures: [...second.state.gpu.textures],
            typed: [...second.state.gpu.typed],
        };
        const skinData = second.state.gpu.buffers.get("skinData");
        expect(skinData).toBeDefined();
        expect(secondDevice.live.has(skinData as GPUBuffer)).toBe(true);

        first.dispose();
        first = undefined;
        expect([...peerResources].every((resource) => secondDevice.live.has(resource))).toBe(true);
        expect([...second.state.gpu.buffers]).toEqual(peerRegistries.buffers);
        expect([...second.state.gpu.textures]).toEqual(peerRegistries.textures);
        expect([...second.state.gpu.typed]).toEqual(peerRegistries.typed);
        expect(hashPhysics(second.state)).toBe(peerHashBeforeDispose);
        expect(readBody(second.state, peerA)).toEqual(peerBodyBeforeDispose);
        expect(secondDevice.live.has(skinData as GPUBuffer)).toBe(true);
        await stepGpuWorld(second.state, "second world after sibling disposal", secondDevice);
        expect(readBody(second.state, peerA)).not.toEqual(peerBodyBeforeDispose);

        second.dispose();
        second = undefined;
        if (!firstPairPixels || !secondPairPixels)
            throw new Error("both paired Worlds must produce a captured frame");
        const firstSoloPixels = await renderAlone(firstDevice, FIRST_CONTENT, "first world");
        const secondSoloPixels = await renderAlone(secondDevice, SECOND_CONTENT, "second world");
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

test("plugin-owned components and GPU paths stay isolated on a shared device", async () => {
    await exerciseIsolationPair(true);
}, 6_000);

test("plugin-owned components and GPU paths stay isolated on separate devices", async () => {
    await exerciseIsolationPair(false);
}, 6_000);
