import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { categories } from "../src/scenes";
import { manualParityScenes } from "./manual-parity";
import { dominoSwitchRanges } from "./switch-dominoes";

setDefaultTimeout(TIMEOUT.node);

test("every Node parity category has a switch file", async () => {
    for (const category of categories) {
        if (category.scenes.every((scene) => manualParityScenes.has(scene.id))) continue;
        expect(
            await Bun.file(
                new URL(`./switch-${category.name.toLowerCase()}.node.ts`, import.meta.url),
            ).text(),
        ).toContain(`testCategorySwitches("${category.name}")`);
    }
});

test("Dominoes switch files cover each of the 600 steps once", async () => {
    const steps: number[] = [];
    for (const [part, range] of Object.entries(dominoSwitchRanges)) {
        expect(
            await Bun.file(new URL(`./switch-dominoes-${part}.node.ts`, import.meta.url)).text(),
        ).toContain(`testDominoSwitches("${part}")`);
        for (let step = range.start; step < range.end; step++) steps.push(step);
    }
    expect(steps).toEqual(Array.from({ length: 600 }, (_, step) => step));
});
