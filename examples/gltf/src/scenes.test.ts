import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import manifest from "../assets.json";
import { categories, modelPath, type Scene, scenes, truckPlacement } from "./scenes";

test("the picker demonstrates each promise once without shrinking the regression corpus", () => {
    const listed = categories.flatMap((category) => category.scenes.map((scene) => scene.id));
    expect(listed.length).toBe(17);
    expect(new Set(listed).size).toBe(listed.length);
    expect(manifest.models).toHaveLength(36);
    expect(
        Object.fromEntries(
            categories.map((category) => [category.name, category.scenes.map((scene) => scene.id)]),
        ),
    ).toEqual({
        animation: ["AnimatedTriangle", "CesiumMan", "SkinnedNormal", "AnimatedMorphCube"],
        geometry: ["MeshPrimitiveModes", "VertexColorTest", "QuantizedMorphCube"],
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
        const written = manifest.written.find((fixture) => fixture.name === scene);
        if (written) {
            expect(modelPath(scene)).toBe(written.path);
            continue;
        }
        expect(
            manifest.models
                .flatMap((model) => model.files)
                .some((file) => modelPath(scene) === `assets/${file.path}`),
        ).toBe(true);
    }
});
test("written corpus fixtures are CC0 and retain their manifest hashes", () => {
    for (const fixture of manifest.written) {
        expect(fixture.license).toBe("CC0-1.0");
        const bytes = readFileSync(new URL(`../public/${fixture.path}`, import.meta.url));
        expect(createHash("sha256").update(bytes).digest("hex")).toBe(fixture.sha256);
    }
});

test("the truck retains its accepted placement", () => {
    expect(truckPlacement).toEqual({ translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] });
});
