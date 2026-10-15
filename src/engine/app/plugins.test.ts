import { afterEach, expect, test } from "bun:test";
import { AudioPlugin } from "../../core/audio";
import { BrowserInputPlugin, createBrowserInputPlugin, InputPlugin } from "../../core/input";
import { MeshPlugin } from "../../core/mesh";
import { PhysicsPlugin } from "../../core/physics";
import { CorePipelinePlugin, RenderingPlugin } from "../../core/rendering";
import { TransformPlugin } from "../../core/transform";
import {
    FogPlugin,
    LinesPlugin,
    OrbitPlugin,
    OutlinePlugin,
    ProfilePlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
    VignettePlugin,
} from "../../extras";
import {
    CharacterPlugin,
    DEFAULT_PLUGINS,
    MeshRenderPlugin,
    StandardAudioPlugin,
    StandardPhysicsPlugin,
    StandardRenderingPlugin,
} from "../../standard";
import { type Plugin, Time } from "../index";
import { createApp } from "./index";

const needsGpu = new Set<Plugin>([
    RenderingPlugin,
    CorePipelinePlugin,
    MeshPlugin,
    MeshRenderPlugin,
    StandardRenderingPlugin,
    FogPlugin,
    LinesPlugin,
    OutlinePlugin,
    ProfilePlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
    VignettePlugin,
]);
const publicPlugins = [
    ...DEFAULT_PLUGINS,
    InputPlugin,
    BrowserInputPlugin,
    createBrowserInputPlugin(),
    MeshPlugin,
    PhysicsPlugin,
    RenderingPlugin,
    CorePipelinePlugin,
    TransformPlugin,
    AudioPlugin,
    StandardAudioPlugin,
    CharacterPlugin,
    FogPlugin,
    LinesPlugin,
    OrbitPlugin,
    OutlinePlugin,
    ProfilePlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
    VignettePlugin,
    StandardPhysicsPlugin,
    MeshRenderPlugin,
    StandardRenderingPlugin,
];
let restoreGpu: (() => void) | undefined;
const apps: Awaited<ReturnType<typeof createApp>>[] = [];

afterEach(() => {
    restoreGpu?.();
    restoreGpu = undefined;
    for (const app of apps.splice(0)) app.dispose();
});

function withoutGpu(): void {
    const previous = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: undefined });
    restoreGpu = () => {
        if (previous) Object.defineProperty(navigator, "gpu", previous);
        else Reflect.deleteProperty(navigator, "gpu");
    };
}

async function verifyPublicPlugin(plugin: Plugin): Promise<void> {
    expect(plugin.gpu !== undefined).toBe(needsGpu.has(plugin));
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        app = await createApp({ defaults: false, plugins: [plugin] });
        apps.push(app);
        if (needsGpu.has(plugin))
            throw new Error(`${plugin.name} unexpectedly built without WebGPU`);
        app.world.step(Time.FIXED_DT);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!needsGpu.has(plugin) || !message.includes("navigator.gpu is missing")) {
            throw new Error(`${plugin.name} failed outside GPU acquisition: ${message}`, {
                cause: error,
            });
        }
    }
}

const uniquePublicPlugins = [...new Set(publicPlugins)];
const deviceIndependentPlugins = uniquePublicPlugins.filter((plugin) => !needsGpu.has(plugin));
const gpuPlugins = uniquePublicPlugins.filter((plugin) => needsGpu.has(plugin));
const PLUGINS_PER_TEST = 4;

for (let start = 0; start < deviceIndependentPlugins.length; start += PLUGINS_PER_TEST) {
    const group = deviceIndependentPlugins.slice(start, start + PLUGINS_PER_TEST);
    test(`device-independent public plugins ${group.map((plugin) => plugin.name).join(", ")} run without WebGPU`, async () => {
        withoutGpu();
        for (const plugin of group) await verifyPublicPlugin(plugin);
    });
}

test("GPU-dependent public plugins refuse without WebGPU at acquisition", async () => {
    withoutGpu();
    for (const plugin of gpuPlugins) await verifyPublicPlugin(plugin);
});
