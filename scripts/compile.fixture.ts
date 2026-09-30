import {
    build,
    CharacterPlugin,
    InputPlugin,
    OrbitPlugin,
    PhysicsPlugin,
    PhysicsProfilePlugin,
    PlayerPlugin,
    ProfilePlugin,
    RenderPlugin,
} from "@dylanebert/shallot";
import { rawDevice } from "../src/engine/runtime";

export const compileSubjects = [
    { name: "engine-only", config: { defaults: false, plugins: [] } },
    { name: "core rendering", config: { defaults: false, plugins: [RenderPlugin] } },
    { name: "Physics", config: { defaults: false, plugins: [PhysicsPlugin] } },
    {
        name: "Physics with core rendering",
        config: { defaults: false, plugins: [PhysicsPlugin, RenderPlugin] },
    },
    {
        name: "Character gameplay",
        config: { defaults: false, plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin] },
    },
    {
        name: "Player gameplay",
        config: { defaults: false, plugins: [PhysicsPlugin, CharacterPlugin, PlayerPlugin] },
    },
    {
        name: "Physics profiling",
        config: { defaults: false, plugins: [PhysicsPlugin, PhysicsProfilePlugin] },
    },
    { name: "Orbit", config: { defaults: false, plugins: [OrbitPlugin] } },
    { name: "profiling", config: { defaults: false, plugins: [ProfilePlugin] } },
    { name: "default plugins", config: { plugins: [] } },
] satisfies { name: string; config: Parameters<typeof build>[0] }[];

export async function measureCompile(config: Parameters<typeof build>[0]) {
    const owner = await build({ defaults: false, plugins: [] });
    const device = rawDevice(owner.state.gpu.device);
    const methods = [
        "createComputePipeline",
        "createComputePipelineAsync",
        "createRenderPipeline",
        "createRenderPipelineAsync",
    ] as const;
    const originals = methods.map((name) => device[name]);
    let pipelines = 0;
    let app: Awaited<ReturnType<typeof build>> | undefined;
    try {
        for (const [index, name] of methods.entries()) {
            Reflect.set(device, name, (...args: unknown[]) => {
                pipelines++;
                return Reflect.apply(originals[index], device, args);
            });
        }
        const start = performance.now();
        app = await build({ ...config, device });
        await device.queue.onSubmittedWorkDone();
        return { pipelines, ms: performance.now() - start };
    } finally {
        app?.dispose();
        for (const [index, name] of methods.entries()) Reflect.set(device, name, originals[index]);
        owner.dispose();
    }
}
