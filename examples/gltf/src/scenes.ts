import type { GltfPlacement } from "@dylanebert/shallot/gltf";

// The picker demonstrates promises; the full asset corpus remains in tests.
export const scenes = {
    TextureSettingsTest: {
        target: [0, -0.58, 0],
        distance: 16,
        yaw: 0,
        pitch: 0,
        promise: "textures",
    },
    AlphaBlendModeTest: {
        target: [0, 1.1, 0],
        distance: 11,
        yaw: 0,
        pitch: 0.1,
        promise: "materials",
    },
    TwoSidedPlane: {
        target: [0, 0, 0],
        distance: 4,
        yaw: 0.7,
        pitch: 0.4,
        promise: "materials",
    },
    OrientationTest: {
        target: [0, 0, 0],
        distance: 20,
        yaw: 0.7,
        pitch: 0.4,
        promise: "transforms",
    },
    NegativeScaleTest: {
        target: [0, 0, 0],
        distance: 17,
        yaw: 0,
        pitch: 0,
        promise: "transforms",
    },
    MeshPrimitiveModes: {
        target: [0, -3, 0],
        distance: 8,
        yaw: 0,
        pitch: 0,
        promise: "geometry",
    },
    MultiUVTest: {
        target: [0, 0, 0],
        distance: 4,
        yaw: 0,
        pitch: 0,
        promise: "textures",
    },
    VertexColorTest: {
        target: [0, 0, 0],
        distance: 5,
        yaw: 0.7,
        pitch: 0.4,
        promise: "geometry",
    },
    TextureCoordinateTest: {
        target: [0, 0, 0],
        distance: 6,
        yaw: 0,
        pitch: 0,
        promise: "textures",
    },
    MetalRoughSpheres: {
        target: [0, 4, 0],
        distance: 40,
        yaw: 0.7,
        pitch: 0.65,
        promise: "materials",
    },
    NormalTangentMirrorTest: {
        target: [0, 0, 0],
        distance: 5,
        yaw: 0,
        pitch: 0,
        promise: "materials",
    },
    CesiumMilkTruck: {
        target: [0, 0, 0],
        distance: 6,
        yaw: 0.7,
        pitch: 0.4,
        promise: "textures",
    },
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
