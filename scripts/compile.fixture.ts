import {
    Camera,
    CharacterPlugin,
    type createApp,
    DEFAULT_PLUGINS,
    EffectStackPlugin,
    FogPlugin,
    InputPlugin,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    OrbitPlugin,
    OutlinePlugin,
    ProfilePlugin,
    RenderingPlugin,
    registerProducerTable,
    ScreenSpaceAmbientOcclusion,
    ScreenSpaceAmbientOcclusionPlugin,
    SpritePlugin,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
} from "@dylanebert/shallot";
import { attachTexture, Views } from "../src/core/rendering";
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
    {
        name: "ScreenSpaceAmbientOcclusion",
        config: { plugins: [ScreenSpaceAmbientOcclusionPlugin] },
    },
    { name: "Outline", config: { plugins: [OutlinePlugin] } },
    { name: "EffectStack", config: { plugins: [EffectStackPlugin] } },
    { name: "Sprite", config: { plugins: [SpritePlugin] } },
] satisfies { name: string; config: Parameters<typeof createApp>[0] }[];

export function initialSsaoCamera(antialias: number): Parameters<typeof createApp>[0] {
    return {
        plugins: [
            ScreenSpaceAmbientOcclusionPlugin,
            {
                name: "InitialSsaoCamera",
                dependencies: [...DEFAULT_PLUGINS, ScreenSpaceAmbientOcclusionPlugin],
                initialize(world) {
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 4, 0] });
                    world.add(camera, Camera, { antialias });
                    world.add(camera, StandardRenderer);
                    world.add(camera, ScreenSpaceAmbientOcclusion);
                    const mesh = world.create();
                    world.add(mesh, Transform);
                    world.add(mesh, MeshInstance);
                    world.add(
                        mesh,
                        MeshMaterial,
                        world.resource(Materials).add(StandardMaterial()),
                    );
                },
            },
        ],
    };
}

export async function measureCompile(
    config: Parameters<typeof createApp>[0],
    onPipeline?: (label: string) => void,
    draw = false,
) {
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
                onPipeline?.(
                    (args[0] as GPUComputePipelineDescriptor | GPURenderPipelineDescriptor).label ??
                        "<unlabelled>",
                );
                return Reflect.apply(originals[index], device, args);
            });
        }
        const start = performance.now();
        app = await createTestGpuApp({ ...config, device });
        await device.queue.onSubmittedWorkDone();
        const warmPipelines = pipelines;
        if (draw) {
            for (const camera of app.world.query([Camera, StandardRenderer])) {
                if (!app.world.resource(Views).has(camera))
                    attachTexture(app.world, camera, { width: 64, height: 64 });
            }
            app.world.step(1 / 60);
            await app.world.frameFence;
        }
        return {
            pipelines,
            latePipelines: pipelines - warmPipelines,
            ms: performance.now() - start,
        };
    } finally {
        app?.dispose();
        for (const [index, name] of methods.entries()) Reflect.set(device, name, originals[index]);
        await disposeGpuApps([owner]);
    }
}
