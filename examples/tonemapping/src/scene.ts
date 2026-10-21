import {
    Camera,
    ColorGrading,
    cube,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    MeshPlugin,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    sphere,
    Tonemapping,
    TonemappingMethod,
    Transform,
} from "@dylanebert/shallot";

import { PatternMaterial, PatternPlugin } from "./pattern";

export const Scene: Resource<{
    camera: number;
    subjects: number[];
    pattern: number;
    meshes: number[];
}> = {
    create: () => ({ camera: -1, subjects: [], pattern: -1, meshes: [] }),
};

export const ScenePlugin: Plugin = {
    name: "TonemappingScene",
    dependencies: [MeshPlugin, StandardRenderingPlugin, PatternPlugin],
    initialize(world) {
        const scene = world.resource(Scene);
        const camera = world.create();
        scene.camera = camera;
        world.add(camera, Camera, { clearColor: 0x141210 });
        world.add(camera, StandardRenderer);
        world.add(camera, Transform, {
            translation: [0.7, 0.7, 1],
            rotation: lookAtRotation(0.7, 0.7, 1, 0, 0.3, 0),
        });
        world.add(camera, ColorGrading);
        world.add(camera, Tonemapping, { method: TonemappingMethod.TonyMcMapface });
        const light = world.create();
        world.add(light, DirectionalLight, { illuminance: 15000, shadowMapsEnabled: 1 });
        const angle = -Math.PI * 0.15;
        const s = Math.sin(angle / 2);
        const c = Math.cos(angle / 2);
        world.add(light, Transform, { rotation: [c * s, s * c, -s * s, c * c] });
        // Ambient lighting replaces Bevy's environment map, which Shallot does not support.
        Object.assign(world.resource(GlobalAmbientLight), { color: 0x8fa8cc, brightness: 80 });
        const meshes = world.resource(Meshes);
        const materials = world.resource(Materials);
        const ball = meshes.add({ name: "Sphere", ...sphere(0.125, 48, 24) });
        const box = meshes.add({ name: "Cube", ...cube([0.25, 0.25, 0.25]) });
        const ground = meshes.add({ name: "Plane", ...cube([100, 0.01, 100]) });
        const subjects: {
            mesh: number;
            position: [number, number, number];
            color: [number, number, number, number];
            roughness: number;
        }[] = [
            { mesh: ground, position: [0, -0.005, 0], color: [0.1, 0.2, 0.1, 1], roughness: 0.5 },
        ];
        // TonemappingTest.gltf's Cube mesh contains five disjoint boxes.
        for (let index = 0; index < 5; index++) {
            subjects.push({
                mesh: box,
                position: [-1 + index * 0.25, 0.125, -index * 0.5],
                color: [1, 1, 1, 1],
                roughness: 0.5,
            });
        }
        const colors: [number, number, number, number][] = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0.2, 0.2, 1, 1],
            [0.2, 1, 0.2, 1],
            [1, 0.2, 0.2, 1],
        ];
        const positions: [number, number, number][] = [
            [-0.55, 0.125, 0.25],
            [-0.3, 0.125, 0],
            [-0.05, 0.125, -0.25],
            [-0.25, 0.125, 0.55],
            [0, 0.125, 0.3],
            [0.25, 0.125, 0.05],
        ];
        colors.forEach((color, index) => {
            subjects.push({ mesh: ball, position: positions[index], color, roughness: 0.089 });
        });
        const pattern = world.create();
        scene.pattern = pattern;
        const rotation = lookAtRotation(0.7, 0.7, 1, 0, 0.3, 0);
        const distance = Math.hypot(0.7, 0.4, 1);
        world.add(pattern, Transform, {
            translation: [0.7 - 0.7 / distance, 0.7 - 0.4 / distance, 1 - 1 / distance],
            rotation,
        });
        world.add(pattern, MeshMaterial, world.resource(PatternMaterial).add({}));
        const patternMesh = meshes.add({
            name: "Test pattern",
            vertices: new Float32Array([
                -0.35, -0.35, 0, 0, 0, 0, 1, 0, 0.35, -0.35, 0, 1, 0, 0, 1, 0, 0.35, 0.35, 0, 1, 0,
                0, 1, 1, -0.35, 0.35, 0, 0, 0, 0, 1, 1,
            ]),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        });
        for (const subject of subjects) {
            const eid = world.create();
            world.add(eid, MeshInstance, { mesh: subject.mesh });
            world.add(
                eid,
                MeshMaterial,
                materials.add(
                    StandardMaterial({
                        baseColor: subject.color,
                        metallic: 0,
                        perceptualRoughness: subject.roughness,
                    }),
                ),
            );
            world.add(eid, Transform, { translation: subject.position });
            scene.subjects.push(eid);
            scene.meshes.push(subject.mesh);
        }
        scene.meshes.push(patternMesh);
    },
};
