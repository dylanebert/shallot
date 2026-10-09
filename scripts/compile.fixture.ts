import {
    CharacterPlugin,
    createApp,
    DEFAULT_PLUGINS,
    InputPlugin,
    OrbitPlugin,
    PlayerPlugin,
    ProfilePlugin,
    RenderingPlugin,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import { rawDevice } from "../src/engine/runtime";
import { gpuRequirements } from "./gpu.fixture";

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
        name: "Player gameplay",
        config: {
            defaults: false,
            plugins: [StandardPhysicsPlugin, CharacterPlugin, PlayerPlugin],
        },
    },
    {
        name: "Physics profiling",
        config: { defaults: false, plugins: [ProfilePlugin, StandardPhysicsPlugin] },
    },
    { name: "Orbit", config: { defaults: false, plugins: [OrbitPlugin] } },
    { name: "profiling", config: { defaults: false, plugins: [ProfilePlugin] } },
    { name: "default plugins", config: { plugins: [] } },
] satisfies { name: string; config: Parameters<typeof createApp>[0] }[];

export async function measureCompile(config: Parameters<typeof createApp>[0]) {
    const plugins = [
        ...DEFAULT_PLUGINS,
        ...compileSubjects.flatMap((subject) => subject.config.plugins),
    ];
    const owner = await createApp({
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
        app = await createApp({ ...config, device });
        await device.queue.onSubmittedWorkDone();
        return { pipelines, ms: performance.now() - start };
    } finally {
        app?.dispose();
        for (const [index, name] of methods.entries()) Reflect.set(device, name, originals[index]);
        owner.dispose();
    }
}
