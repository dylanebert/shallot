import { expect, test } from "bun:test";
import {
    CellsPlugin,
    FogPlugin,
    GltfPlugin,
    LinesPlugin,
    liveSkin,
    OrbitOverlayPlugin,
    OrbitPlugin,
    OutlinePlugin,
    PhysicsProfilePlugin,
    PlayerPlugin,
    ProfilePlugin,
    Skin,
    SkinPlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
} from "../../extras";
import { Color, DEFAULT_PLUGINS, PartPlugin } from "../../standard";
import { AudioPlugin } from "../../transitional/audio";
import { BvhPlugin } from "../../transitional/bvh";
import { CharacterPlugin } from "../../transitional/character";
import { MirrorPlugin } from "../../transitional/mirror";
import {
    Body,
    hash as hashPhysics,
    PhysicsPlugin,
    physicsWorld,
    readBody,
    restore as restorePhysics,
    ShapeKind,
    Spring,
    setVelocity,
    snapshot as snapshotPhysics,
} from "../../transitional/physics";
import type { World } from "../../transitional/physics/api";
import { Compute, type Plugin, type State, Time } from "../index";
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
await peer.setupGlobals();

async function trackedDevice() {
    const adapter = await navigator.gpu.requestAdapter();
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
    const device = await adapter.requestDevice({
        requiredFeatures: ["bgra8unorm-storage", "rg11b10ufloat-renderable", "timestamp-query"],
        requiredLimits,
    });
    const live = new Set<GPUBuffer | GPUTexture>();
    const createBuffer = device.createBuffer.bind(device);
    const createTexture = device.createTexture.bind(device);
    Object.defineProperties(device, {
        createBuffer: {
            configurable: true,
            writable: true,
            value: (descriptor: GPUBufferDescriptor) => {
                const buffer = createBuffer(descriptor);
                live.add(buffer);
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
                const destroy = texture.destroy.bind(texture);
                texture.destroy = () => {
                    if (live.delete(texture)) destroy();
                };
                return texture;
            },
        },
    });
    return { device, live };
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

function skinSeedPlugin(): Plugin {
    return {
        name: "GpuIsolationSkinSeed",
        dependencies: [PartPlugin, SkinPlugin],
        initialize(state) {
            const eid = state.create();
            state.add(eid, Color);
            state.add(eid, Skin);
            const skin = liveSkin(state);
            Skin.anim.x.set(eid, skin.alloc(eid, 1, state.stamp(eid)));
            skin.flush(Compute.device);
        },
    };
}

test("every exported plugin isolates two live worlds through snapshot, restore and disposal", async () => {
    const firstDevice = await trackedDevice();
    const secondDevice = await trackedDevice();
    const seed = skinSeedPlugin();
    let first: Awaited<ReturnType<typeof build>> | undefined;
    let second: Awaited<ReturnType<typeof build>> | undefined;
    let firstWorld: World | null = null;
    let saved: Uint8Array | undefined;
    try {
        first = await build({
            defaults: false,
            plugins: [...everyPlugin, seed],
            device: firstDevice.device,
        });
        const firstA = addBody(first.state, 2);
        const firstB = addBody(first.state, 3);
        addSpring(first.state, firstA, firstB);

        let armed = false;
        let peerState: State | undefined;
        const firstSnapshot = { hash: 0n, body: null as ReturnType<typeof readBody> };
        const probe: Plugin = {
            name: "GpuIsolationPhysicsProbe",
            systems: [
                {
                    name: "GpuIsolationPhysicsProbe",
                    group: "fixed",
                    update(state) {
                        if (!armed || state !== peerState) return;
                        armed = false;
                        const peerBody = readBody(state, peerBodyEid);
                        const peerHash = hashPhysics(state);
                        expect(readBody(first!.state, firstA)).toEqual(firstSnapshot.body);
                        expect(hashPhysics(first!.state)).toBe(firstSnapshot.hash);

                        firstWorld = physicsWorld(first!.state);
                        saved = snapshotPhysics(first!.state);
                        expect(() => restorePhysics(state, saved!)).toThrow(
                            "physics: unknown world snapshot",
                        );
                        setVelocity(first!.state, firstA, 7, 0, 0);
                        expect(readBody(first!.state, firstA)?.vel[0]).toBeCloseTo(7);
                        restorePhysics(first!.state, saved!);

                        expect(hashPhysics(first!.state)).toBe(firstSnapshot.hash);
                        expect(readBody(first!.state, firstA)).toEqual(firstSnapshot.body);
                        expect(hashPhysics(state)).toBe(peerHash);
                        expect(readBody(state, peerBodyEid)).toEqual(peerBody);
                        expect(physicsWorld(first!.state)?.getCounters().jointCount).toBe(1);
                        expect(physicsWorld(state)?.getCounters().jointCount).toBe(1);
                    },
                },
            ],
        };
        second = await build({
            defaults: false,
            plugins: [...everyPlugin, seed, probe],
            device: secondDevice.device,
        });
        const peerA = addBody(second.state, 20);
        const peerBodyEid = addBody(second.state, 21);
        addSpring(second.state, peerA, peerBodyEid);
        peerState = second.state;

        for (let i = 0; i < 8; i++) {
            first.state.step(Time.FIXED_DT);
            second.state.step(Time.FIXED_DT);
        }
        expect(physicsWorld(first.state)?.getCounters().jointCount).toBe(1);
        expect(physicsWorld(second.state)?.getCounters().jointCount).toBe(1);
        firstSnapshot.hash = hashPhysics(first.state);
        firstSnapshot.body = readBody(first.state, firstA);
        if (!firstSnapshot.body) throw new Error("first Physics body did not become live");

        armed = true;
        second.state.step(Time.FIXED_DT);
        expect(armed).toBe(false);

        const peerHashBeforeDispose = hashPhysics(second.state);
        const peerBodyBeforeDispose = readBody(second.state, peerA);
        const peerResources = new Set(secondDevice.live);
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
        second.state.step(Time.FIXED_DT);
        expect(readBody(second.state, peerA)?.pos[1]).toBeLessThan(
            peerBodyBeforeDispose?.pos[1] ?? 0,
        );

        const oldWorld = firstWorld as unknown as {
            restore(snapshot: Uint8Array): void;
        } | null;
        expect(oldWorld).not.toBeNull();
        expect(() => oldWorld!.restore(saved!)).toThrow("physics: unknown world snapshot");
    } finally {
        second?.dispose();
        first?.dispose();
    }
    expect(firstDevice.live.size).toBe(0);
    expect(secondDevice.live.size).toBe(0);
}, 60_000);
