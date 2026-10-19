import { expect, test } from "bun:test";
import { categories, scenes } from "./scenes";

test("each picker label selects its scene in its category, exactly once", () => {
    expect(categories).toEqual([
        {
            name: "Joints",
            scenes: [
                { id: "bridge", name: "Bridge" },
                { id: "filter", name: "Filter" },
                { id: "parallel", name: "Parallel Spring" },
                { id: "prismatic", name: "Prismatic" },
                { id: "revolute", name: "Revolute" },
                { id: "wheel", name: "Wheel" },
            ],
        },
        { name: "Stacking", scenes: [{ id: "boxStack", name: "Box Stack" }] },
    ]);
    expect(
        categories.flatMap((category) => category.scenes.map((scene): string => scene.id)).sort(),
    ).toEqual(Object.keys(scenes).sort());
});
