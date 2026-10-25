import {
    Camera,
    cube,
    DirectionalLight,
    GlobalAmbientLight,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    ScreenSpaceAmbientOcclusion,
    ScreenSpaceAmbientOcclusionPlugin,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin, ProfilePlugin } from "@dylanebert/shallot/extras";

export const Scene: Resource<{ camera: number; cubes: number[]; sphere: number }> = {
    create: () => ({ camera: -1, cubes: [], sphere: -1 }),
};

export function setOcclusion(world: World, enabled: boolean): void {
    const camera = world.resource(Scene).camera;
    if (enabled) world.add(camera, ScreenSpaceAmbientOcclusion);
    else world.remove(camera, ScreenSpaceAmbientOcclusion);
}

export const ScenePlugin: Plugin = {
    name: "AmbientOcclusionScene",
    dependencies: [ScreenSpaceAmbientOcclusionPlugin, OrbitPlugin, ProfilePlugin],
    initialize(world) {
        const scene = world.resource(Scene);
        const camera = world.create();
        scene.camera = camera;
        world.add(camera, Camera, { clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        world.add(camera, Orbit, {
            pan: [0, 0, 0],
            distance: Math.sqrt(12),
            yaw: Math.atan2(-2, -2),
            pitch: Math.atan2(2, Math.hypot(2, 2)),
        });
        world.add(camera, Transform);
        setOcclusion(world, true);
        const sun = world.create();
        world.add(sun, DirectionalLight, {
            color: 0xffe4c4,
            illuminance: 15000,
            shadowMapsEnabled: 1,
        });
        const s = Math.sin((-0.15 * Math.PI) / 2);
        const c = Math.cos((-0.15 * Math.PI) / 2);
        world.add(sun, Transform, { rotation: [c * s, s * c, -s * s, c * c] });
        Object.assign(world.resource(GlobalAmbientLight), { color: 0x8fa8cc, brightness: 80 });
        const meshes = world.resource(Meshes);
        const materials = world.resource(Materials);
        const box = meshes.add({ name: "Unit cube", ...cube([1, 1, 1]) });
        const ball = meshes.add({ name: "Sphere", ...sphere(0.4, 72, 36) });
        const carbon = materials.add(
            StandardMaterial({ baseColor: [0.15, 0.14, 0.125, 1], perceptualRoughness: 1 }),
        );
        const accent = materials.add(
            StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 1 }),
        );
        for (const translation of [
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ] as const) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [...translation] });
            world.add(eid, MeshInstance, { mesh: box });
            world.add(eid, MeshMaterial, carbon);
            scene.cubes.push(eid);
        }
        const eid = world.create();
        scene.sphere = eid;
        world.add(eid, Transform);
        world.add(eid, MeshInstance, { mesh: ball });
        world.add(eid, MeshMaterial, accent);
    },
    systems: [
        {
            name: "moveSphere",
            group: "fixed",
            update(world) {
                world
                    .storage(Transform)
                    .translation.y.set(
                        world.resource(Scene).sphere,
                        Math.sin(world.time.elapsed / 1.7) * 0.7,
                    );
            },
        },
    ],
};
