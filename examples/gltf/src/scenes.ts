import type { GltfPlacement } from "@dylanebert/shallot/gltf";

// glTF-Sample-Assets model-index.json at assets.json's revision; tags overlap.
export const scenes = {
    Box: {
        target: [0, 0, 0],
        distance: 3,
        ground: -0.51,
        yaw: 0.7,
        pitch: 0.4,
        tags: ["core", "testing"],
    },
    BoxTextured: {
        target: [0, 0, 0],
        distance: 3,
        ground: -0.51,
        yaw: 0.7,
        pitch: 0.4,
        tags: ["core", "issues", "testing"],
    },
    TextureSettingsTest: {
        target: [0, -0.58, 0],
        distance: 16,
        ground: -5.63,
        yaw: 0,
        pitch: 0,
        tags: ["core", "testing"],
    },
    AlphaBlendModeTest: {
        target: [0, 1.1, 0],
        distance: 11,
        ground: -0.11,
        yaw: 0,
        pitch: 0.1,
        tags: ["core", "testing"],
    },
    TwoSidedPlane: {
        target: [0, 0, 0],
        distance: 4,
        ground: -0.1,
        yaw: 0.7,
        pitch: 0.4,
        tags: ["core"],
    },
    OrientationTest: {
        target: [0, 0, 0],
        distance: 20,
        ground: -5.34,
        yaw: 0.7,
        pitch: 0.4,
        tags: ["core", "testing"],
    },
    SimpleMeshes: {
        target: [1, 0.5, 0],
        distance: 4,
        ground: -0.01,
        yaw: 0,
        pitch: 0.1,
        tags: ["core", "testing", "written"],
    },
    TriangleWithoutIndices: {
        target: [0.5, 0.5, 0],
        distance: 3,
        ground: -0.01,
        yaw: 0,
        pitch: 0.1,
        tags: ["core", "testing"],
    },
    CesiumMilkTruck: {
        target: [0, 0, 0],
        distance: 6,
        ground: -0.6,
        yaw: 0.7,
        pitch: 0.4,
        tags: ["core", "issues", "testing"],
    },
} as const;
export type Scene = keyof typeof scenes;
export const categories = ["core", "issues", "testing", "written"].map((name) => ({
    name,
    scenes: (Object.keys(scenes) as Scene[])
        .filter((id) => (scenes[id].tags as readonly string[]).includes(name))
        .map((id) => ({ id, name: id })),
}));
export const modelPath = (scene: Scene) => `assets/Models/${scene}/glTF/${scene}.gltf`;
export const truckPlacement: GltfPlacement = { translation: [0, -0.5, 0], scale: [0.7, 0.7, 0.7] };
