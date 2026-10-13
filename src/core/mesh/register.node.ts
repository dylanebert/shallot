import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { createTestGpuApp, expectAdapterNotice, gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import type { createApp, Plugin, System, World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { classifyAdapter } from "../../engine/runtime/adapter";
import {
    MeshPreprocessSystem,
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import {
    AmbientLight,
    attachTexture,
    BeginFrameSystem,
    Camera,
    captureTexture,
} from "../rendering";
import { Transform } from "../transform";
import {
    type Mesh,
    Meshes,
    type MeshHandle,
    MeshInstance,
    MeshPlugin,
    registerMesh,
} from "./index";

setDefaultTimeout(CEILING.node);

// a 4×4 quad facing +z, wider than the default view from z = 5
const vertices = new Float32Array([
    -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0, 0, 0, 0, 1,
    1,
]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
const register = (world: World) => registerMesh(world, { name: "slab", vertices, indices });
function handleForLabel(world: World, label: string): MeshHandle {
    for (const [handle, mesh] of world.resource(Meshes).entries())
        if (mesh.name === label) return handle;
    throw new Error(`no mesh labelled "${label}"`);
}
// four draw systems, each after the last, ending before the pack, as a producer writing its instances does
function chainedDrawSystems(): System[] {
    const systems: System[] = [];
    let previous: System = BeginFrameSystem;
    for (let i = 0; i < 4; i++) {
        previous = {
            group: "draw",
            after: [previous],
            before: i === 3 ? [MeshPreprocessSystem] : [],
            update() {},
        };
        systems.push(previous);
    }
    return systems;
}
const place = (world: World, mesh: MeshHandle) => {
    const slab = world.create();
    world.add(slab, Transform);
    world.add(slab, MeshInstance, { mesh });
};
const arms = {
    "during initialize": {
        name: "InitializeSlab",
        dependencies: [MeshPlugin],
        initialize(world) {
            register(world);
        },
    },
    "in warm": {
        name: "WarmSlab",
        dependencies: [MeshPlugin],
        warm(world) {
            register(world);
        },
    },
    "after createApp": { name: "LateSlab", dependencies: [MeshPlugin] },
    "after createApp behind four chained draw systems": {
        name: "ChainedSlab",
        dependencies: [MeshPlugin],
        systems: chainedDrawSystems(),
    },
    "in a system": {
        name: "SystemSlab",
        dependencies: [MeshPlugin],
        systems: [
            {
                update(world) {
                    if (world.resource(Meshes).size > 1) return;
                    place(world, register(world));
                },
            },
        ],
    },
    "in a draw-group system": {
        name: "DrawSlab",
        dependencies: [MeshPlugin],
        systems: [
            {
                group: "draw",
                update(world) {
                    if (world.resource(Meshes).size > 1) return;
                    place(world, register(world));
                },
            },
        ],
    },
    "directly without streams": { name: "BareSlab", dependencies: [MeshPlugin] },
    reinitialized: {
        name: "ReinitializedSlab",
        dependencies: [MeshPlugin],
        initialize(world) {
            register(world);
        },
    },
} satisfies Record<string, Plugin>;
type Arm = keyof typeof arms;
const order = Object.keys(arms) as Arm[];
const late = ["in warm", "after createApp", "in a system"] as const;

const subjects = gpuApps(
    import.meta.path,
    order.map((arm) => ({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, arms[arm]],
    })),
);

const chained = "after createApp behind four chained draw systems";
let chainedValidation: GPUError | null = null;
const frames = new Map<Arm, Uint8ClampedArray>();
async function frame(arm: Arm, steps = 1): Promise<Uint8ClampedArray> {
    const cached = frames.get(arm);
    if (cached) return cached;
    const { world } = subjects()[order.indexOf(arm)];
    if (arm.startsWith("after createApp")) register(world);
    if (arm === "directly without streams") {
        const meshes = world.resource(Meshes);
        const [, cube] = [...meshes.entries()].find(([, mesh]) => mesh.name === "cube")!;
        meshes.register({ ...cube, name: "slab", position: undefined, quant: undefined });
    }
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: 998.09574 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    if (!arm.endsWith("system")) place(world, handleForLabel(world, "slab"));
    if (arm === "reinitialized") await reinitialize(world);
    if (arm === chained) world.gpu.device.pushErrorScope("validation");
    for (let i = 0; i < steps; i++) world.step(0);
    const { rgba } = await captureTexture(world, camera);
    if (arm === chained) chainedValidation = await world.gpu.device.popErrorScope();
    frames.set(arm, rgba);
    return rgba;
}

test("a mesh registered during initialize draws on the first frame", async () => {
    const rgba = await frame("during initialize");
    const pixel = (x: number, y: number) =>
        Array.from(rgba.subarray((y * 32 + x) * 4, (y * 32 + x) * 4 + 3));
    expect(pixel(16, 16)).not.toEqual(pixel(0, 0));
});

for (const arm of late) {
    test(`a mesh registered ${arm} draws on the next frame as one registered during initialize`, async () => {
        expect(await frame(arm)).toEqual(await frame("during initialize"));
    });
}

test("a mesh registered after createApp behind four chained draw systems before the pack draws on the next frame as one registered during initialize", async () => {
    expect(await frame(chained)).toEqual(await frame("during initialize"));
    expect(chainedValidation).toBeNull();
});

test("a mesh registered in a draw-group system draws on the following frame without a warning", async () => {
    const rgba = await frame("in a draw-group system", 2);
    expect(rgba).toEqual(await frame("during initialize"));
});

test("a mesh registered directly without quantized streams is skipped with a warning", async () => {
    expectWarning(
        /^standard: draw "mesh:material:StandardMaterial:slab:1" skipped — mesh "slab" has no quantized position\/quant stream$/,
    );
    expectWarning(
        /^standard: draw "mesh:material:VertexMaterial:slab:1" skipped — mesh "slab" has no quantized position\/quant stream$/,
    );
    await frame("directly without streams");
});

// Reinitialization makes each new family replace the previous one on the next step.
const families: Mesh[] = [];
let validation: GPUError | null = null;
async function reinitialize(world: World): Promise<void> {
    families.push(world.resource(Meshes).get(handleForLabel(world, "slab"))!);
    world.gpu.device.pushErrorScope("validation");
    for (let i = 0; i < 3; i++) {
        await MeshPlugin.initialize!(world);
        register(world);
        world.step(0);
        families.push(world.resource(Meshes).get(handleForLabel(world, "slab"))!);
    }
    await world.gpu.device.queue.onSubmittedWorkDone();
    validation = await world.gpu.device.popErrorScope();
}

test("three reruns of MeshPlugin.initialize destroy every family but the live one", async () => {
    const rgba = await frame("reinitialized");
    expect(validation).toBeNull();
    expect(
        families.map((m) => [m.vertices, m.position!, m.quant!, m.indices].map((b) => b.destroyed)),
    ).toEqual([
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [false, false, false, false],
    ]);
    expect(rgba).toEqual(await frame("during initialize"));
});

test("a mesh registered in setup, before MeshPlugin initializes, is refused by name", async () => {
    const owner = subjects()[0];
    // The build below supplies a device without its adapter, which acquisition stamps unidentified
    // whatever the owner's adapter is (src/engine/runtime/gpu.ts, requestGPU).
    expectAdapterNotice(classifyAdapter({ present: true }));
    const device = rawDevice(owner.world.gpu.device);
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        app = await createTestGpuApp({
            defaults: false,
            plugins: [MeshPlugin],
            setup: register,
            device,
        });
    } catch (error) {
        expect(String(error)).toContain('"slab"');
        expect(String(error)).toContain("MeshPlugin");
        return;
    } finally {
        app?.dispose();
    }
    throw new Error("setup registration was not refused");
});
