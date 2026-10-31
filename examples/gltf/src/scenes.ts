import type { GltfPlacement } from "@dylanebert/shallot/gltf";

// The picker demonstrates promises; the full asset corpus remains in tests.
export const scenes = {
    TextureSettingsTest: { promise: "textures" },
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
} as const;
export type Scene = keyof typeof scenes;
export const categories = ["geometry", "transforms", "materials", "textures"].map((name) => ({
    name,
    scenes: (Object.keys(scenes) as Scene[])
        .filter((id) => scenes[id].promise === name)
        .map((id) => ({ id, name: id })),
}));
export const modelPath = (scene: Scene) => `assets/Models/${scene}/glTF/${scene}.gltf`;
export const truckPlacement: GltfPlacement = { translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] };
