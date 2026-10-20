import {
    Body,
    BodyType,
    Camera,
    Character,
    composeMat4,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    type Resource,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { RenderLayers } from "@dylanebert/shallot/rendering";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const Scene: Resource<{
    camera: number;
    modelCamera: number;
    player: number;
    light: number;
    parts: number[];
    matrix: Float32Array;
}> = {
    create: () => ({
        camera: 0,
        modelCamera: 0,
        player: 0,
        light: 0,
        parts: [],
        matrix: new Float32Array(16),
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
// Frame-authored attachments follow the camera after its interpolated position and look are settled.
// Writing them on fixed ticks would give meshes tick history that the camera does not use.
export const ViewModelSystem = {
    name: "view-model",
    group: "simulation" as const,
    after: [UpdatePlayerControlSystem],
    update(world: World) {
        const s = world.resource(Scene),
            t = world.storage(Transform);
        const p = t.translation.column,
            q = t.rotation.column,
            offset = s.camera * 4;
        const x = p[offset]!,
            y = p[offset + 1]!,
            z = p[offset + 2]!;
        const qx = q[offset]!,
            qy = q[offset + 1]!,
            qz = q[offset + 2]!,
            qw = q[offset + 3]!;
        t.translation.set(s.modelCamera, x, y, z);
        t.rotation.set(s.modelCamera, qx, qy, qz, qw);
        const m = composeMat4(x, y, z, qx, qy, qz, qw, 1, 1, 1, s.matrix);
        for (const [index, eid] of s.parts.entries()) {
            const [px, py, pz] = PARTS[index]!.position;
            t.translation.set(
                eid,
                m[0]! * px + m[4]! * py + m[8]! * pz + x,
                m[1]! * px + m[5]! * py + m[9]! * pz + y,
                m[2]! * px + m[6]! * py + m[10]! * pz + z,
            );
            t.rotation.set(eid, qx, qy, qz, qw);
        }
    },
};
export const ScenePlugin: Plugin = {
    name: "ViewModel",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin],
    systems: [ViewModelSystem],
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
        world.add(s.camera, Transform, { translation: [0, 1.6, 2] });
        s.modelCamera = world.create();
        world.add(s.modelCamera, Camera, { fov: 70, order: 1 });
        world.add(s.modelCamera, StandardRenderer);
        world.add(s.modelCamera, Transform);
        world.add(s.modelCamera, RenderLayers, { mask: 2 });
        s.player = world.create();
        world.add(s.player, Transform, { translation: [0, 0.9, 2] });
        world.add(s.player, Body, { type: BodyType.Kinematic });
        world.add(s.player, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.55, 0],
            capsuleCenter2: [0, 0.55, 0],
            radius: 0.3,
        });
        world.add(s.player, Character);
        world.add(s.player, Player, { camera: s.camera });
        world.add(s.player, LocalPlayer);
        const surfaces = [
            { position: [0, -0.1, 0], size: [10, 0.2, 12], color: [0.085, 0.08, 0.072, 1] },
            { position: [0, 1.5, -1], size: [3, 3, 0.2], color: [0.11, 0.105, 0.095, 1] },
        ] as const;
        for (const surface of surfaces) {
            const eid = block(world, surface.position, surface.size, surface.color);
            world.add(eid, Body, { type: BodyType.Static });
            world.add(eid, Shape);
        }
        s.parts = PARTS.map((p) => {
            const eid = block(world, p.position, p.size, p.color);
            world.add(eid, RenderLayers, { mask: 2 });
            world.add(eid, NotShadowCaster);
            return eid;
        });
        ViewModelSystem.update(world);
    },
};
