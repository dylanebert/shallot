import {
    CharacterPlugin,
    type createApp,
    DEFAULT_PLUGINS,
    FogPlugin,
    InputPlugin,
    Materials,
    Meshes,
    OrbitPlugin,
    OutlinePlugin,
    ProfilePlugin,
    RenderingPlugin,
    registerProducerTable,
    SpritePlugin,
    StandardMaterial,
    StandardPhysicsPlugin,
    VignettePlugin,
} from "@dylanebert/shallot";
import { rawDevice } from "../src/engine/runtime";
import { createTestGpuApp, disposeGpuApps, gpuRequirements } from "./gpu.fixture";

export const compileSubjects = [
    { name: "engine-only", config: { defaults: false, plugins: [] } },
    { name: "core rendering", config: { defaults: false, plugins: [RenderingPlugin] } },
    { name: "Physics", config: { defaults: false, plugins: [StandardPhysicsPlugin] } },
    {
        name: "Physics with core rendering",
        config: { defaults: false, plugins: [StandardPhysicsPlugin, RenderingPlugin] },
    },
    {
        name: "Character gameplay",
        config: { defaults: false, plugins: [StandardPhysicsPlugin, CharacterPlugin, InputPlugin] },
    },
    {
        name: "Physics profiling",
        config: { defaults: false, plugins: [ProfilePlugin, StandardPhysicsPlugin] },
    },
    { name: "Orbit", config: { defaults: false, plugins: [OrbitPlugin] } },
    { name: "profiling", config: { defaults: false, plugins: [ProfilePlugin] } },
    { name: "default plugins", config: { plugins: [] } },
    {
        name: "producer table",
        config: {
            plugins: [
                {
                    name: "CompileProducer",
                    warm(world) {
                        registerProducerTable(world, {
                            name: "compile-producer",
                            capacity: 1,
                            meshes: [[...world.resource(Meshes).entries()][0]![0]],
                            materials: [world.resource(Materials).add(StandardMaterial())],
                        });
                    },
                },
            ],
        },
    },
    { name: "Fog", config: { plugins: [FogPlugin] } },
    { name: "Outline", config: { plugins: [OutlinePlugin] } },
    { name: "Vignette", config: { plugins: [VignettePlugin] } },
    { name: "Sprite", config: { plugins: [SpritePlugin] } },
] satisfies { name: string; config: Parameters<typeof createApp>[0] }[];

export async function measureCompile(config: Parameters<typeof createApp>[0]) {
    const plugins = [
        ...DEFAULT_PLUGINS,
        ...compileSubjects.flatMap((subject) => subject.config.plugins),
    ];
    const owner = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "CompileProbeDevice", gpu: gpuRequirements(plugins) }],
    });
    const device = rawDevice(owner.world.gpu.device);
    const methods = [
        "createComputePipeline",
        "createComputePipelineAsync",
        "createRenderPipeline",
        "createRenderPipelineAsync",
    ] as const;
    const originals = methods.map((name) => device[name]);
    let pipelines = 0;
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        for (const [index, name] of methods.entries()) {
            Reflect.set(device, name, (...args: unknown[]) => {
                pipelines++;
                return Reflect.apply(originals[index], device, args);
            });
        }
        const start = performance.now();
        app = await createTestGpuApp({ ...config, device });
        await device.queue.onSubmittedWorkDone();
        return { pipelines, ms: performance.now() - start };
    } finally {
        app?.dispose();
        for (const [index, name] of methods.entries()) Reflect.set(device, name, originals[index]);
        await disposeGpuApps([owner]);
    }
}
