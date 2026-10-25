import { DiagnosticsStore, ScreenSpaceAmbientOcclusion, type World } from "@dylanebert/shallot";
import { Scene } from "./scene";

export const settings = [
    { field: "radius", label: "Radius", min: 0.1, max: 8, step: 0.0001, value: 0.7285 },
    {
        field: "constantObjectThickness",
        label: "Thickness",
        min: 0.0625,
        max: 4,
        step: 0.0625,
        value: 0.25,
    },
] as const;

export function setSetting(world: World, index: number, value: number): void {
    world
        .storage(ScreenSpaceAmbientOcclusion)
        [settings[index].field].set(world.resource(Scene).camera, value);
}
export function setQuality(world: World, quality: number): void {
    world
        .storage(ScreenSpaceAmbientOcclusion)
        .qualityLevel.set(world.resource(Scene).camera, quality);
}
export function cost(world: World): string {
    const store = world.resource(DiagnosticsStore);
    if (store.get("render/timestamp_query")?.value() !== 1) return "—";
    if (!world.has(world.resource(Scene).camera, ScreenSpaceAmbientOcclusion)) return "0.00 ms";
    let total = 0;
    for (const diagnostic of store.iter()) {
        const path = diagnostic.path.path;
        if (path.startsWith("render/standard:ssao-") && path.endsWith("/elapsed_gpu"))
            total += diagnostic.value() ?? 0;
    }
    return `${total.toFixed(2)} ms`;
}
