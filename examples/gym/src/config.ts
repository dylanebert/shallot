import { traits, resource, createCone } from "@dylanebert/shallot";

export { createCone };

let _coneMeshId = -1;
export function coneMeshId(): number {
    return _coneMeshId;
}
export function setConeMeshId(id: number) {
    _coneMeshId = id;
}

export const Lorenz = {
    x: [] as number[],
    y: [] as number[],
    z: [] as number[],
};
traits(Lorenz, { defaults: () => ({ x: 0, y: 0, z: 0 }) });

export const Benchmark = {
    count: [] as number[],
};
traits(Benchmark, { defaults: () => ({ count: 100 }) });

export const PipelineMode = { Raster: 0, Raytracing: 1 } as const;
export const CameraMode = { Static: 0, Pan: 1 } as const;
export const Layout = { Lorenz: 0, Grid: 1 } as const;

export const BenchConfig = {
    pipeline: [] as number[],
    effects: [] as number[],
    flags: [] as number[],
    camera: [] as number[],
    layout: [] as number[],
};
traits(BenchConfig, {
    defaults: () => ({
        pipeline: PipelineMode.Raster,
        effects: 0,
        flags: 0,
        camera: CameraMode.Static,
        layout: Layout.Lorenz,
    }),
});

interface Spawned {
    eid: number;
    baseX: number;
}

export interface SpawnedLight {
    eid: number;
    bulb: number;
    basePos: [number, number, number];
}

interface BenchSnap {
    pipeline: number;
    effects: number;
    flags: number;
    layout: number;
}

export interface BenchmarkState {
    spawned: Spawned[];
    spawnedLights: SpawnedLight[];
    prev: BenchSnap;
    vertexSurface: number;
    fragmentSurface: number;
    trivialSurface: number;
    externalSpawner: boolean;
}

export const BenchmarkState = resource<BenchmarkState>("benchmark");

export type Pipeline = "raster" | "raytracing";
export const PIPELINES: Pipeline[] = ["raster", "raytracing"];
export type CameraModeName = "static" | "pan";
export type LayoutName = "lorenz" | "grid";
