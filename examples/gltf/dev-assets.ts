import manifest from "./assets.json";
import { modelPath, type Scene, scenes } from "./src/scenes";

export function pickerModels(paths: Readonly<Record<string, string>>) {
    const models = new Set<(typeof manifest.models)[number]>();
    for (const [scene, path] of Object.entries(paths)) {
        const model = manifest.models.find((model) =>
            model.files.some((file) => `assets/${file.path}` === path),
        );
        if (model) models.add(model);
        else if (!manifest.written.some((fixture) => fixture.path === path))
            throw new Error(`${scene}: no manifest model or written fixture for ${path}`);
    }
    return [...models];
}

export const pickerPaths = Object.fromEntries(
    (Object.keys(scenes) as Scene[]).map((scene) => [scene, modelPath(scene)]),
);
export const devModels = pickerModels(pickerPaths);
