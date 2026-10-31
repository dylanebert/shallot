import { expect, test } from "bun:test";
import manifest from "../assets.json";
import { categories, modelPath, type Scene, scenes, truckPlacement } from "./scenes";

test("the picker demonstrates each promise once without shrinking the regression corpus", () => {
    const listed = categories.flatMap((category) => category.scenes.map((scene) => scene.id));
    expect(listed.length).toBe(12);
    expect(new Set(listed).size).toBe(listed.length);
    expect(manifest.models).toHaveLength(18);
    expect(
        Object.fromEntries(
            categories.map((category) => [category.name, category.scenes.map((scene) => scene.id)]),
        ),
    ).toEqual({
        geometry: ["MeshPrimitiveModes", "VertexColorTest"],
        transforms: ["OrientationTest", "NegativeScaleTest"],
        materials: [
            "AlphaBlendModeTest",
            "TwoSidedPlane",
            "MetalRoughSpheres",
            "NormalTangentMirrorTest",
        ],
        textures: [
            "TextureSettingsTest",
            "MultiUVTest",
            "TextureCoordinateTest",
            "CesiumMilkTruck",
        ],
    });
    for (const scene of Object.keys(scenes) as Scene[]) {
        expect(
            manifest.models
                .find((model) => model.name === scene)!
                .files.some((file) => modelPath(scene) === `assets/${file.path}`),
        ).toBe(true);
    }
});
test("the truck retains its accepted placement", () => {
    expect(truckPlacement).toEqual({ translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] });
});
