import {
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    ViewportPlugin,
    RenderPlugin,
    RasterPlugin,
    PhysicsPlugin,
    AudioPlugin,
    PlayerPlugin,
    Physics,
    Body,
    Transform,
    Compute,
} from "@dylanebert/shallot";
import type { Config, Plugin, State } from "@dylanebert/shallot";
import {
    OrbitPlugin,
    RaytracingPlugin,
    OutlinePlugin,
    AcousticPlugin,
    TextPlugin,
    LinesPlugin,
    ArrowsPlugin,
    StatsPlugin,
    BloomPlugin,
    LensFlarePlugin,
    GodRaysPlugin,
} from "@dylanebert/shallot/extras";
import { BenchmarkPlugin } from "./benchmark";
import { parseEffects } from "./effects";
import {
    buildPhysicsScenarioPlugin,
    buildRenderTestPlugin,
    buildPhysicsTestPlugin,
    buildAudioPlugin,
    buildPlayerPlugin,
    createCrosshair,
} from "./scenarios";
import type {
    PhysicsTestVariant,
    AudioRoom,
    RenderTestLighting,
    RenderTestShape,
    RenderTestVariant,
} from "./scenarios";
import { getCapabilities } from "./capabilities";
import type { GymCapabilities } from "./capabilities";
import { wireBridge } from "./bridge";
import { fireInit } from "./state.svelte";

const BASE_PLUGINS: Plugin[] = [
    TransformsPlugin,
    InputPlugin,
    ComputePlugin,
    ViewportPlugin,
    RenderPlugin,
];

const EXTRA_PLUGINS: Record<string, Plugin> = {
    raytracing: RaytracingPlugin,
    orbit: OrbitPlugin,
    physics: PhysicsPlugin,
    benchmark: BenchmarkPlugin,
};

export { createMeasure } from "./measure";
export {
    namesToConfig,
    CAMERA_EFFECTS,
    ENV_EFFECTS,
} from "./effects";
export {
    Benchmark,
    BenchConfig,
    PipelineMode,
    CameraMode,
    Layout,
    PIPELINES,
} from "./config";
export type {
    Pipeline,
    CameraModeName,
    LayoutName,
} from "./config";
export {
    setRenderTestShape,
    setRenderTestVariant,
    setRenderTestLighting,
    setPhysicsTestVariant,
    setPileShapes,
    RENDER_TEST_SHAPES,
    RENDER_TEST_VARIANTS,
    RENDER_TEST_LIGHTING,
    PHYSICS_TEST_VARIANTS,
    PILE_SHAPES,
    AUDIO_ROOMS,
    stepCount,
} from "./scenarios";
export type {
    RenderTestShape,
    RenderTestVariant,
    RenderTestLighting,
    PhysicsTestVariant,
    PileShape,
    AudioRoom,
} from "./scenarios";

const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "");
const scenarioName = params.get("scenario");

export const noUI = params.has("noui");

export const urlPipeline = params.get("pipeline") as "raster" | "raytracing" | null;
export const urlEffects = parseEffects(params.get("effects"));

export const urlLighting = params.get("lighting") as RenderTestLighting | null;
export const urlShape = params.get("shape") as RenderTestShape | null;
export const urlVariant = params.get("variant") as RenderTestVariant | null;
export const urlLayout = (params.get("layout") ?? "lorenz") as "lorenz" | "grid";
export const urlCamera = (params.get("camera") ?? "static") as "static" | "pan";
export const urlCount = params.has("count")
    ? parseInt(params.get("count")!, 10) || undefined
    : undefined;
export const urlRamp = params.has("ramp");
export const urlShapes = params.has("shapes")
    ? params
          .get("shapes")!
          .split(",")
          .map((s) => parseInt(s, 10) as 0 | 1 | 2 | 3)
          .filter((n) => n >= 0 && n <= 3)
    : undefined;
export const urlHeight = params.has("height") ? parseFloat(params.get("height")!) || 0 : 0;
export const urlText = params.has("text");
export const urlArrow = params.has("arrow");
export const urlPhysicsTest = params.get("test") as PhysicsTestVariant | null;
export const urlAudioRoom = params.get("room") as AudioRoom | null;
export const urlSources = params.has("sources") ? parseInt(params.get("sources")!, 10) || 1 : 1;
export const activeScenarioName: string | null = scenarioName;
export const capabilities: GymCapabilities = getCapabilities(scenarioName);

function parseExtraPlugins(): Plugin[] | null {
    const raw = params.get("plugins");
    if (!raw) return null;
    if (raw === "none") return [];
    return raw.split(",").flatMap((name) => {
        const plugin = EXTRA_PLUGINS[name.trim()];
        if (!plugin) console.warn(`Unknown plugin: ${name}`);
        return plugin ? [plugin] : [];
    });
}

function pipelinePlugins(): Plugin[] {
    if (urlPipeline === "raytracing") return [RaytracingPlugin];
    return [RasterPlugin];
}

function buildConfig(): Config {
    const pluginOverride = parseExtraPlugins();

    let scene: string | undefined;
    let scenario: Plugin | undefined;

    if (scenarioName === "audio") {
        scenario = buildAudioPlugin(urlAudioRoom ?? "living", urlSources);
    } else if (scenarioName === "player") {
        scenario = buildPlayerPlugin();
    } else if (scenarioName === "render") {
        scenario = buildRenderTestPlugin();
    } else if (scenarioName === "physics") {
        const test = (params.get("test") ?? "box") as PhysicsTestVariant;
        scenario = buildPhysicsTestPlugin(test);
    } else if (scenarioName) {
        scenario = buildPhysicsScenarioPlugin(urlCount ?? 100, urlRamp, urlShapes, urlHeight);
    } else {
        scene = "/scenes/benchmark.scene";
    }

    const isPhysicsTest = scenarioName === "physics";
    const isRenderTest = scenarioName === "render";
    const extras =
        pluginOverride ??
        (urlRamp
            ? [
                  ...(capabilities.firstPerson ? [PlayerPlugin] : [OrbitPlugin]),
                  RasterPlugin,
                  RaytracingPlugin,
                  PhysicsPlugin,
                  AudioPlugin,
                  AcousticPlugin,
                  OutlinePlugin,
                  BloomPlugin,
                  LensFlarePlugin,
                  GodRaysPlugin,
                  TextPlugin,
                  LinesPlugin,
                  ArrowsPlugin,
              ]
            : [
                  ...(capabilities.firstPerson ? [PlayerPlugin] : [OrbitPlugin]),
                  ...pipelinePlugins(),
                  ...(capabilities.physics ? [PhysicsPlugin] : []),
                  ...(capabilities.audio ? [AudioPlugin, AcousticPlugin] : []),
                  ...(isPhysicsTest || scenarioName === "player" ? [OutlinePlugin] : []),
                  BloomPlugin,
                  LensFlarePlugin,
                  GodRaysPlugin,
                  ...(isRenderTest ? [TextPlugin, LinesPlugin, ArrowsPlugin] : []),
              ]);
    if (!extras.includes(BenchmarkPlugin)) extras.push(BenchmarkPlugin);
    extras.push(StatsPlugin);
    if (scenario) extras.push(scenario);

    return {
        defaults: false,
        plugins: [...BASE_PLUGINS, ...extras],
        ...(scene && { scene }),
        ...(scenarioName === "player" && {
            ui: (container: HTMLElement) => createCrosshair(container),
        }),
        setup(state: State) {
            wireBridge(
                state,
                {
                    urlPipeline,
                    urlEffects,
                    urlLighting,
                    urlShape,
                    urlVariant,
                    urlLayout,
                    urlCamera,
                    urlCount,
                    urlText,
                    urlArrow,
                    capabilities,
                },
                fireInit,
            );
        },
    };
}

export const config: Config = buildConfig();
export { Physics, Body, Transform, Compute };
