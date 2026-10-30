import { expect, test } from "bun:test";
import manifest from "../assets.json";
import { categories, modelPath, type Scene, scenes, truckPlacement } from "./scenes";

test("every synced model appears under exactly its upstream catalog tags", () => {
    expect(Object.keys(scenes).sort()).toEqual(manifest.models.map((model) => model.name).sort());
    expect(
        Object.fromEntries(
            categories.map((category) => [category.name, category.scenes.map((scene) => scene.id)]),
        ),
    ).toEqual({
        core: [
            "Box",
            "BoxTextured",
            "TextureSettingsTest",
            "AlphaBlendModeTest",
            "TwoSidedPlane",
            "OrientationTest",
            "SimpleMeshes",
            "TriangleWithoutIndices",
            "CesiumMilkTruck",
        ],
        issues: ["BoxTextured", "CesiumMilkTruck"],
        testing: [
            "Box",
            "BoxTextured",
            "TextureSettingsTest",
            "AlphaBlendModeTest",
            "OrientationTest",
            "SimpleMeshes",
            "TriangleWithoutIndices",
            "CesiumMilkTruck",
        ],
        written: ["SimpleMeshes"],
    });
    for (const scene of Object.keys(scenes) as Scene[]) {
        expect(
            manifest.models
                .find((model) => model.name === scene)!
                .files.some((file) => modelPath(scene) === `assets/${file.path}`),
        ).toBe(true);
        expect(scenes[scene].distance).toBeGreaterThan(0);
    }
});
test("the truck retains its accepted placement and orbit framing", () => {
    expect(truckPlacement).toEqual({ translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] });
    expect(scenes.CesiumMilkTruck).toMatchObject({
        distance: 6,
        yaw: 0.7,
        pitch: 0.4,
        ground: -0.6,
        target: [0, 0, 0],
    });
});
