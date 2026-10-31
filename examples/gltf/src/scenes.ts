import type { GltfPlacement } from "@dylanebert/shallot/gltf";

// The picker demonstrates promises; the full asset corpus remains in tests.
export const scenes = {
    TextureSettingsTest: { promise: "textures" },
    TextureTransformMultiTest: { promise: "textures" },
    EmissiveStrengthTest: { promise: "materials" },
    AlphaBlendModeTest: { promise: "materials" },
    TwoSidedPlane: { promise: "materials" },
    OrientationTest: { promise: "transforms" },
    NegativeScaleTest: { promise: "transforms" },
    MeshPrimitiveModes: { promise: "geometry" },
    MultiUVTest: { promise: "textures" },
    VertexColorTest: { promise: "geometry" },
    TextureCoordinateTest: { promise: "textures" },
    MetalRoughSpheres: { promise: "materials" },
    NormalTangentMirrorTest: { promise: "materials" },
    CesiumMilkTruck: { promise: "textures" },
    AnimatedTriangle: { promise: "animation" },
    CesiumMan: { promise: "animation" },
    SkinnedNormal: { promise: "animation" },
    AnimatedMorphCube: { promise: "animation" },
    QuantizedMorphCube: { promise: "geometry" },
} as const;
export type Scene = keyof typeof scenes;
export const categories = ["geometry", "transforms", "materials", "textures", "animation"].map(
    (name) => ({
        name,
        scenes: (Object.keys(scenes) as Scene[])
            .filter((id) => scenes[id].promise === name)
            .map((id) => ({ id, name: id })),
    }),
);
export const modelPath = (scene: Scene) =>
    scene === "QuantizedMorphCube"
        ? "assets/Models/AnimatedMorphCube/glTF-Quantized/AnimatedMorphCube.gltf"
        : scene === "SkinnedNormal"
          ? "models/SkinnedNormal.gltf"
          : `assets/Models/${scene}/glTF/${scene}.gltf`;
export const truckPlacement: GltfPlacement = { translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] };
