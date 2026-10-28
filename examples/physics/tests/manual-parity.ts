import { categories, type Scene } from "../src/scenes";

/** Scenes whose parity and switching live in the manual oracles. */
export const manualParityScenes: ReadonlySet<Scene> = new Set([
    ...categories
        .find((category) => category.name === "Benchmark")!
        .scenes.map((scene) => scene.id),
    "village",
    "meshCreation",
]);
