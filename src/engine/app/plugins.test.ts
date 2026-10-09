import { afterEach, expect, test } from "bun:test";
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
    PlayerPlugin,
    ProfilePlugin,
    SkyPlugin,
    SpritePlugin,
    TextPlugin,
    VignettePlugin,
} from "../../extras";
import {
    AudioPlugin,
    CharacterPlugin,
    DEFAULT_PLUGINS,
    MeshRenderPlugin,
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
    CharacterPlugin,
    FogPlugin,
    LinesPlugin,
    OrbitPlugin,
    OutlinePlugin,
    PlayerPlugin,
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

test("each public plugin runs without a device or refuses its missing device at acquisition", async () => {
    withoutGpu();
    for (const plugin of new Set(publicPlugins)) {
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
});
