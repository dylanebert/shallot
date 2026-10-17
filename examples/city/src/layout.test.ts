import { expect, test } from "bun:test";
import { layout, POPULATION, SEED } from "./layout";

test("the same seed repeats placements and a different seed changes them", () => {
    const placements = layout(SEED, POPULATION);
    expect(layout(SEED, POPULATION)).toEqual(placements);
    expect(layout(SEED + 1, POPULATION)).not.toEqual(placements);
});

test("the requested population uses both kit parts and exactly the six material indices", () => {
    for (const population of [128, 256, POPULATION]) {
        const placements = layout(SEED, population);
        expect(placements).toHaveLength(population);
        expect(new Set(placements.map((p) => p.part))).toEqual(new Set(["block", "dome"]));
        expect(new Set(placements.map((p) => p.material))).toEqual(new Set([0, 1, 2, 3, 4, 5]));
    }
});
