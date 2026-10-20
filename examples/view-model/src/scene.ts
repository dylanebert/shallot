import {
    Camera,
    Devices,
    DirectionalLight,
    GlobalAmbientLight,
    InputPlugin,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    type Resource,
    requirePointerLock,
    StandardMaterial,
    StandardRenderer,
    Time,
    Transform,
    type World,
} from "@dylanebert/shallot";

import { RenderLayers } from "@dylanebert/shallot/rendering";

export const Scene: Resource<{
    camera: number;
    modelCamera: number;
    light: number;
    wall: number;
    parts: number[];
    held: boolean;
    z: number;
    yaw: number;
    pitch: number;
    drop: boolean;
    move: number;
}> = {
    create: () => ({
        camera: 0,
        modelCamera: 0,
        light: 0,
        wall: 0,
        parts: [],
        held: true,
        z: 2,
        yaw: 0,
        pitch: 0,
        drop: false,
        move: 0,
    }),
};
const PARTS = [
    { position: [0.25, -0.36, -0.72], size: [0.12, 0.3, 0.18], color: [0.15, 0.14, 0.125, 1] },
    { position: [0.25, -0.22, -0.8], size: [0.26, 0.16, 0.45], color: [0.72, 0.4, 0.05, 1] },
    { position: [0.25, -0.12, -1.1], size: [0.13, 0.16, 0.3], color: [0.042, 0.039, 0.035, 1] },
] as const;
function block(
    world: World,
    position: readonly [number, number, number],
    size: readonly [number, number, number],
    color: readonly [number, number, number, number],
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale: size });
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: color })),
    );
    return eid;
}
export function toggleItem(world: World) {
    const scene = world.resource(Scene);
    scene.held = !scene.held;
    for (const eid of scene.parts) {
        world.storage(RenderLayers).mask.set(eid, scene.held ? 2 : 1);
        if (!scene.held) world.remove(eid, NotShadowCaster);
        else world.add(eid, NotShadowCaster);
    }
}
export function pose(world: World) {
    const s = world.resource(Scene);
    const sy = Math.sin(s.yaw / 2),
        cy = Math.cos(s.yaw / 2);
    const sp = Math.sin(s.pitch / 2),
        cp = Math.cos(s.pitch / 2);
    const rotation = [cy * sp, sy * cp, -sy * sp, cy * cp];
    for (const eid of [s.camera, s.modelCamera]) {
        world.storage(Transform).translation.set(eid, 0, 1.6, s.z);
        world
            .storage(Transform)
            .rotation.set(eid, ...(rotation as [number, number, number, number]));
    }
    if (!s.held) return;
    for (const [i, eid] of s.parts.entries()) {
        const [x, y, z] = PARTS[i]!.position;
        const py = y * Math.cos(s.pitch) - z * Math.sin(s.pitch);
        const pz = y * Math.sin(s.pitch) + z * Math.cos(s.pitch);
        world
            .storage(Transform)
            .translation.set(
                eid,
                x * Math.cos(s.yaw) + pz * Math.sin(s.yaw),
                1.6 + py,
                s.z - x * Math.sin(s.yaw) + pz * Math.cos(s.yaw),
            );
        world
            .storage(Transform)
            .rotation.set(eid, ...(rotation as [number, number, number, number]));
    }
}
export const ScenePlugin: Plugin = {
    name: "ViewModel",
    dependencies: [InputPlugin],
    initialize(world) {
        const s = world.resource(Scene);
        world.resource(GlobalAmbientLight).color = 0xd8e4ff;
        world.resource(GlobalAmbientLight).brightness = 380;
        s.light = world.create();
        world.add(s.light, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, 0) });
        world.add(s.light, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 6000,
            shadowMapsEnabled: 1,
        });
        world.add(s.light, RenderLayers, { mask: 3 });
        s.camera = world.create();
        world.add(s.camera, Camera, { fov: 90, clearColor: 0x191816 });
        world.add(s.camera, StandardRenderer);
        world.add(s.camera, Transform);
        s.modelCamera = world.create();
        world.add(s.modelCamera, Camera, { fov: 70, order: 1 });
        world.add(s.modelCamera, StandardRenderer);
        world.add(s.modelCamera, Transform);
        world.add(s.modelCamera, RenderLayers, { mask: 2 });
        block(world, [0, -0.1, 0], [10, 0.2, 12], [0.085, 0.08, 0.072, 1]);
        s.wall = block(world, [0, 1.5, -1], [3, 3, 0.2], [0.11, 0.105, 0.095, 1]);
        block(world, [-2.1, 0.5, -2], [0.7, 1, 0.7], [0.06, 0.057, 0.052, 1]);
        s.parts = PARTS.map((p) => {
            const eid = block(world, p.position, p.size, p.color);
            world.add(eid, RenderLayers, { mask: 2 });
            world.add(eid, NotShadowCaster);
            return eid;
        });
        pose(world);
    },
    systems: [
        {
            name: "sample",
            group: "setup",
            update(world) {
                const s = world.resource(Scene),
                    d = world.resource(Devices);
                s.move = Number(d.keys.held.has("KeyS")) - Number(d.keys.held.has("KeyW"));
                if (d.keys.pressed.has("KeyE")) s.drop = true;
                if (d.pointer.lock.status === "locked") {
                    s.yaw -= d.pointer.deltaX * 0.002;
                    s.pitch = Math.max(-1.4, Math.min(1.4, s.pitch - d.pointer.deltaY * 0.002));
                }
            },
        },
        {
            name: "move",
            group: "fixed",
            update(world) {
                const s = world.resource(Scene);
                s.z = Math.max(-0.65, Math.min(4, s.z + s.move * 2 * Time.FIXED_DT));
                if (s.drop) {
                    toggleItem(world);
                    s.drop = false;
                }
                pose(world);
            },
        },
        {
            name: "look",
            group: "simulation",
            setup(world) {
                requirePointerLock(world, true);
            },
            update: pose,
        },
    ],
};
