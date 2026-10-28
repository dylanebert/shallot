import {
    Camera,
    DebandDither,
    DirectionalLight,
    GlobalAmbientLight,
    ImageFilterMode,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    Resolution,
    registerImage,
    StandardMaterial,
    StandardRenderer,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { Posterize, PosterizePlugin } from "./posterize";
export const Scene = { create: () => ({ camera: -1 }) };
export const ScenePlugin: Plugin = {
    name: "LowResScene",
    dependencies: [OrbitPlugin, PosterizePlugin],
    initialize(world) {
        Object.assign(world.resource(GlobalAmbientLight), { color: 0xd8e4ff, brightness: 380 });
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 6000,
            shadowMapsEnabled: 1,
        });
        const camera = world.create();
        world.resource(Scene).camera = camera;
        world.add(camera, Camera, { clearColor: 0x191816, antialias: 0 });
        world.add(camera, Resolution, { height: 360 });
        world.add(camera, DebandDither, { enabled: 0 });
        world.add(camera, Posterize);
        world.add(camera, StandardRenderer);
        world.add(camera, Transform);
        world.add(camera, Orbit, { pan: [0, 0.7, 0], distance: 11, yaw: 0.65, pitch: 0.55 });
        const materials = world.resource(Materials);
        const carbon = materials.add(
            StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
        );
        const stone = materials.add(
            StandardMaterial({ baseColor: [0.15, 0.14, 0.125, 1], perceptualRoughness: 0.8 }),
        );
        const image = registerImage(world, new URL("../public/crate.png", import.meta.url).href, {
            sampler: {
                magFilter: ImageFilterMode.Nearest,
                minFilter: ImageFilterMode.Nearest,
                mipmapFilter: ImageFilterMode.Nearest,
            },
        });
        const crate = materials.add(
            StandardMaterial({ baseColorTexture: image, perceptualRoughness: 0.8 }),
        );
        const blocks: {
            position: [number, number, number];
            scale: [number, number, number];
            material: typeof carbon;
        }[] = [
            { position: [0, -0.3, 0], scale: [6, 0.6, 5], material: carbon },
            { position: [-1.7, 0.2, -1.2], scale: [1.8, 0.4, 1.4], material: stone },
            { position: [-1.5, 1.1, -1.2], scale: [1.3, 1.4, 1.2], material: crate },
            { position: [0.2, 0.7, 0.3], scale: [1.4, 1.4, 1.4], material: crate },
            { position: [1.6, 0.5, -0.8], scale: [1, 1, 1], material: crate },
            { position: [0.2, 1.9, 0.3], scale: [1, 1, 1], material: crate },
        ];
        for (const { position, scale, material } of blocks) {
            const eid = world.create();
            world.add(eid, MeshInstance);
            world.add(eid, MeshMaterial, material);
            world.add(eid, Transform, { translation: position, scale });
        }
    },
};
