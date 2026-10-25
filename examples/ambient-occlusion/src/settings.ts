import { ScreenSpaceAmbientOcclusion, type World } from "@dylanebert/shallot";
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
