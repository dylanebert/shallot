import type { World } from "@dylanebert/shallot";
import { ChromaticAberration, LensDistortion, Vignette } from "@dylanebert/shallot/extras";
import { Scene } from "./scene";

export const settings = [
    [
        { field: "intensity", label: "Intensity", min: -1, max: 1, value: 0.5 },
        { field: "x", label: "Horizontal", min: 0, max: 1, value: 1 },
        { field: "y", label: "Vertical", min: 0, max: 1, value: 1 },
    ],
    [{ field: "intensity", label: "Intensity", min: 0, max: 0.4, value: 0.02 }],
    [
        { field: "intensity", label: "Intensity", min: 0, max: 1, value: 0.65 },
        { field: "radius", label: "Radius", min: 0, max: 2, value: 0.9 },
        // Bevy bounds these only below; ten includes its defaults and the useful falloff range.
        { field: "smoothness", label: "Smoothness", min: 0.01, max: 10, value: 2 },
        { field: "roundness", label: "Roundness", min: 0.01, max: 10, value: 1 },
        { field: "edgeCompensation", label: "Edge compensation", min: 0, max: 1, value: 1 },
    ],
] as const;

export function setSetting(world: World, effect: number, index: number, value: number): void {
    const camera = world.resource(Scene).camera;
    if (effect === 0) {
        const storage = world.storage(LensDistortion);
        if (index === 0) storage.intensity.set(camera, value);
        else if (index === 1) storage.multiplier.x.set(camera, value);
        else storage.multiplier.y.set(camera, value);
    } else if (effect === 1) {
        const storage = world.storage(ChromaticAberration);
        storage.intensity.set(camera, value);
        storage.maxSamples.set(
            camera,
            Math.round(Math.min(64, Math.max(8, ((value - 0.02) / 0.18) * 56 + 8))),
        );
    } else {
        world.storage(Vignette)[settings[2][index].field].set(camera, value);
    }
}
