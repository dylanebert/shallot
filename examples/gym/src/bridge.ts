import { Camera, Raytracing, Physics, readBodies } from "@dylanebert/shallot";
import type { State, System } from "@dylanebert/shallot";

const MAX_CAPACITY = 1 << 20;
import { BenchConfig, Benchmark, PipelineMode, CameraMode, Layout } from "./config";
import type { Pipeline, CameraModeName, LayoutName } from "./config";
import { namesToConfig } from "./effects";
import { createMeasure } from "./measure";
import {
    setRenderTestShape,
    setRenderTestVariant,
    setRenderTestLighting,
    setRenderTestText,
    setRenderTestArrow,
    setPileShapes,
    setAudioRoom,
    setPhysicsTestVariant,
} from "./scenarios";
import type {
    RenderTestShape,
    RenderTestVariant,
    RenderTestLighting,
    PileShape,
    AudioRoom,
    PhysicsTestVariant,
} from "./scenarios";
import type { GymCapabilities } from "./capabilities";

export interface GymInit {
    pipeline: Pipeline;
    objectCount: number;
    maxCapacity: number;
}

export interface BridgeOpts {
    urlPipeline: Pipeline | null;
    urlEffects: string[];
    urlLighting: RenderTestLighting | null;
    urlShape: RenderTestShape | null;
    urlVariant: RenderTestVariant | null;
    urlLayout: LayoutName;
    urlCamera: CameraModeName;
    urlCount: number | undefined;
    urlText: boolean;
    urlArrow: boolean;
    capabilities: GymCapabilities;
}

export function syncParam(key: string, value: string | null): void {
    const p = new URLSearchParams(location.search);
    if (value === null || value === "") {
        p.delete(key);
    } else {
        p.set(key, value);
    }
    const qs = p.toString();
    history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
}

function toPipelineMode(p: Pipeline): number {
    if (p === "raytracing") return PipelineMode.Raytracing;
    return PipelineMode.Raster;
}

export function setPipeline(state: State, p: Pipeline): void {
    const cam = state.only([Camera]);
    if (cam >= 0) BenchConfig.pipeline[cam] = toPipelineMode(p);
    syncParam("pipeline", p);
}

export function setCameraMode(state: State, mode: CameraModeName): void {
    const cam = state.only([Camera]);
    if (cam >= 0) BenchConfig.camera[cam] = mode === "pan" ? CameraMode.Pan : CameraMode.Static;
    syncParam("camera", mode);
}

export function setLayout(state: State, mode: LayoutName): void {
    const cam = state.only([Camera]);
    if (cam >= 0) BenchConfig.layout[cam] = mode === "grid" ? Layout.Grid : Layout.Lorenz;
    syncParam("layout", mode);
}

export function setCount(state: State, n: number): void {
    const count = Math.max(1, Math.min(MAX_CAPACITY, n));
    const benchEid = state.only([Benchmark]);
    if (benchEid >= 0) Benchmark.count[benchEid] = count;
    syncParam("count", String(count));
}

export function setEffects(state: State, names: string[]): void {
    const cam = state.only([Camera]);
    if (cam < 0) return;
    const { effects, flags } = namesToConfig(names);
    BenchConfig.effects[cam] = effects;
    BenchConfig.flags[cam] = flags;
    syncParam("effects", names.length > 0 ? names.join(",") : null);
}

export function setRenderShape(shape: RenderTestShape): void {
    setRenderTestShape(shape);
    syncParam("shape", shape);
}

export function setRenderVariant(variant: RenderTestVariant): void {
    setRenderTestVariant(variant);
    syncParam("variant", variant);
}

export function setRenderLighting(mode: RenderTestLighting): void {
    setRenderTestLighting(mode);
    syncParam("lighting", mode);
}

export function setRenderText(on: boolean): void {
    setRenderTestText(on);
}

export function setRenderArrow(on: boolean): void {
    setRenderTestArrow(on);
}

export function setRoom(room: AudioRoom): void {
    setAudioRoom(room);
    syncParam("room", room);
}

export function setPhysicsTest(name: PhysicsTestVariant): void {
    setPhysicsTestVariant(name);
    syncParam("test", name);
}

export function setBridgePileShapes(shapes: PileShape[]): void {
    setPileShapes(shapes);
    syncParam("shapes", shapes.join(","));
}

export function wireBridge(state: State, opts: BridgeOpts, onInit: (data: GymInit) => void): void {
    const bridgeSystem: System = {
        group: "draw",
        setup(s: State) {
            const cam = s.only([Camera]);
            const benchEid = s.only([Benchmark]);

            let pipeline: Pipeline = "raster";
            if (opts.urlPipeline) {
                pipeline = opts.urlPipeline;
            } else if (cam >= 0 && s.hasComponent(cam, Raytracing)) {
                pipeline = "raytracing";
            }

            if (cam >= 0) {
                BenchConfig.pipeline[cam] = toPipelineMode(pipeline);
                BenchConfig.layout[cam] = opts.urlLayout === "grid" ? Layout.Grid : Layout.Lorenz;
                BenchConfig.camera[cam] =
                    opts.urlCamera === "pan" ? CameraMode.Pan : CameraMode.Static;
            }

            if (opts.urlEffects.length > 0 && cam >= 0) {
                const { effects, flags } = namesToConfig(opts.urlEffects);
                BenchConfig.effects[cam] = effects;
                BenchConfig.flags[cam] = flags;
            }

            if (opts.capabilities.dynamicCount && opts.urlCount !== undefined && benchEid >= 0) {
                Benchmark.count[benchEid] = opts.urlCount;
            }

            if (opts.urlShape) setRenderTestShape(opts.urlShape);
            if (opts.urlVariant) setRenderTestVariant(opts.urlVariant);
            if (opts.urlText) setRenderTestText(true);
            if (opts.urlArrow) setRenderTestArrow(true);

            if (opts.urlLighting) {
                if (opts.capabilities.renderTestShapes) {
                    setRenderTestLighting(opts.urlLighting);
                } else if (cam >= 0) {
                    const effectSet = new Set(opts.urlEffects);
                    for (const name of ["nosun", "pl1", "pl2", "pl3", "pl4"]) {
                        effectSet.delete(name);
                    }
                    const mode = opts.urlLighting;
                    if (mode === "point" || mode === "multipoint") effectSet.add("nosun");
                    if (mode === "point" || mode === "dir+pt") effectSet.add("pl1");
                    if (mode === "multipoint") {
                        effectSet.add("pl1");
                        effectSet.add("pl2");
                        effectSet.add("pl3");
                        effectSet.add("pl4");
                    }
                    const { effects, flags } = namesToConfig([...effectSet]);
                    BenchConfig.effects[cam] = effects;
                    BenchConfig.flags[cam] = flags;
                }
            }

            const objectCount = benchEid >= 0 ? Benchmark.count[benchEid] : 0;
            onInit({ pipeline, objectCount, maxCapacity: MAX_CAPACITY });

            window.__benchmark = {
                ready: true,
                measure: createMeasure(s),
                setEffects: (names: string[]) => {
                    if (cam < 0) return;
                    const { effects, flags } = namesToConfig(names);
                    BenchConfig.effects[cam] = effects;
                    BenchConfig.flags[cam] = flags;
                },
                setCount: (n: number) => {
                    const count = Math.max(1, Math.min(MAX_CAPACITY, n));
                    if (benchEid >= 0) Benchmark.count[benchEid] = count;
                },
                setCamera: (mode: CameraModeName) => {
                    if (cam >= 0)
                        BenchConfig.camera[cam] =
                            mode === "pan" ? CameraMode.Pan : CameraMode.Static;
                },
                setLayout: (mode: LayoutName) => {
                    if (cam >= 0)
                        BenchConfig.layout[cam] = mode === "grid" ? Layout.Grid : Layout.Lorenz;
                },
                setRenderTestShape: (name: RenderTestShape) => setRenderTestShape(name),
                setRenderTestVariant: (name: RenderTestVariant) => setRenderTestVariant(name),
                setRenderTestLighting: (name: RenderTestLighting) => setRenderTestLighting(name),
                setRoom: (room: AudioRoom) => {
                    setAudioRoom(room);
                    syncParam("room", room);
                },
                setPipeline: (p: Pipeline) => {
                    if (cam >= 0) BenchConfig.pipeline[cam] = toPipelineMode(p);
                },
                setPhysicsTestVariant: (name: string) => {
                    setPhysicsTestVariant(name as PhysicsTestVariant);
                    syncParam("test", name);
                },
                readBodies: async () => {
                    const gpu = Physics.from(s);
                    if (!gpu) return [];
                    return readBodies(gpu);
                },
            };
        },
    };

    state.register(bridgeSystem);
}
