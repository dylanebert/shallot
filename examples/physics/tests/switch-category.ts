import { test } from "bun:test";
import { categories } from "../src/scenes";
import { manualParityScenes } from "./manual-parity";
import { expectSwitchesEqualFresh } from "./switch";

export function testCategorySwitches(name: string): void {
    const category = categories.find((category) => category.name === name);
    if (!category) throw new Error(`Unknown scene category: ${name}`);
    test(`switching ${name} equals fresh scenes through 600 steps after a changed, grabbed Bridge`, () =>
        expectSwitchesEqualFresh(
            category.scenes
                .map((scene) => scene.id)
                // Bridge and Dominoes have separate files; manual scenes switch in their oracle.
                // Creation Benchmark's unstepped build frame is held by mesh-inspector.node.ts.
                .filter(
                    (scene) =>
                        scene !== "bridge" &&
                        scene !== "dominoes" &&
                        scene !== "meshCreation" &&
                        !manualParityScenes.has(scene),
                ),
        ));
}
