import type { World } from "@dylanebert/shallot";
import { Bloom } from "@dylanebert/shallot/bloom";
import { Scene } from "./scene";

export const settings = [
    { field: "intensity", label: "Intensity", min: 0, max: 1, value: 0.15 },
    { field: "lowFrequencyBoost", label: "Low-frequency boost", min: 0, max: 1, value: 0.7 },
    { field: "lowFrequencyBoostCurvature", label: "Boost curvature", min: 0, max: 1, value: 0.95 },
    { field: "highPassFrequency", label: "High-pass frequency", min: 0.01, max: 1, value: 1 },
    { field: "threshold", label: "Threshold", min: 0, max: 100, value: 0 },
    { field: "thresholdSoftness", label: "Threshold softness", min: 0, max: 1, value: 0 },
    { field: "scale", label: "Horizontal scale", min: 0, max: 8, value: 1 },
] as const;

export function setSetting(world: World, index: number, value: number): void {
    const storage = world.storage(Bloom);
    const camera = world.resource(Scene).camera;
    const field = settings[index].field;
    if (field === "scale") storage.scale.x.set(camera, value);
    else storage[field].set(camera, value);
}

export function setMode(world: World, mode: number): void {
    world.storage(Bloom).compositeMode.set(world.resource(Scene).camera, mode);
}
