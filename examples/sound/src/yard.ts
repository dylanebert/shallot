import {
    Body,
    BodyType,
    Camera,
    Character,
    DirectionalLight,
    GlobalAmbientLight,
    Listener,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    Shape,
    ShapeKind,
    Sound,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { loadClip, StandardAudioPlugin } from "@dylanebert/shallot/standard/audio";
import { LocalPlayer, Player, PlayerPlugin } from "./player";
import { InteractionInput, Sliding, SoundGamePlugin, SpeakerPart, Usable } from "./sound";

export const Yard: Resource<{ camera: number; player: number; speaker: number; bell: number }> = {
    create: () => ({ camera: 0, player: 0, speaker: 0, bell: 0 }),
};
function block(
    world: World,
    position: readonly [number, number, number],
    size: readonly [number, number, number],
    color: readonly [number, number, number, number],
): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale: size });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: color })),
    );
    return eid;
}
export const YardPlugin: Plugin = {
    name: "Yard",
    dependencies: [StandardPhysicsPlugin, StandardAudioPlugin, PlayerPlugin, SoundGamePlugin],
    async initialize(world) {
        const music = await loadClip(world, "/Project_Utopia.ogg");
        const chime = await loadClip(world, "/bong_001.ogg");
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = 300;
        const sun = world.create();
        world.add(sun, Transform, { rotation: [-0.38, -0.2, 0, 0.903] });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 18000,
            shadowMapsEnabled: 1,
        });
        block(world, [0, -0.25, 0], [40, 0.5, 40], [0.085, 0.08, 0.072, 1]);
        for (const x of [-12, 12])
            block(world, [x, 0.5, 0], [0.3, 1, 24], [0.042, 0.039, 0.035, 1]);
        for (const z of [-12, 12])
            block(world, [0, 0.5, z], [24, 1, 0.3], [0.042, 0.039, 0.035, 1]);
        // The speaker itself owns its Sound and moving placement.
        const speaker = block(world, [0, 1.4, -2], [0.9, 1.6, 0.65], [0.04, 0.43, 0.53, 1]);
        world.storage(Body).type.set(speaker, BodyType.Kinematic);
        world.add(speaker, Sound, { clip: music, loop: 1, volume: 0.7 });
        world.add(speaker, Sliding);
        world.add(speaker, Usable);
        const ball = world.resource(Meshes).add({ name: "round", ...sphere() });
        for (const [height, diameter] of [
            [1.1, 0.62],
            [1.85, 0.34],
        ] as const) {
            const cone = world.create();
            world.add(cone, Transform, {
                translation: [0, height, -1.65],
                scale: [diameter, diameter, 0.1],
            });
            world.add(cone, MeshInstance, { mesh: ball });
            world.add(
                cone,
                MeshMaterial,
                world
                    .resource(Materials)
                    .add(StandardMaterial({ baseColor: [0.042, 0.039, 0.035, 1] })),
            );
            world.add(cone, SpeakerPart, { speaker });
        }
        const foot = block(world, [0, 0.3, -2], [0.25, 0.6, 0.25], [0.042, 0.039, 0.035, 1]);
        world.remove(foot, Body);
        world.remove(foot, Shape);
        world.add(foot, SpeakerPart, { speaker });
        block(world, [0, 0.015, -2], [7.2, 0.03, 0.08], [0.658, 0.301, 0.117, 1]);
        const bell = block(world, [5, 1.4, -2], [0.65, 0.65, 0.65], [0.72, 0.4, 0.05, 1]);
        world.add(bell, Usable, { bell: 1, clip: chime });
        world.storage(MeshInstance).mesh.set(bell, ball);
        const rim = block(world, [5, 1.12, -2], [0.85, 0.12, 0.85], [0.72, 0.4, 0.05, 1]);
        world.storage(MeshInstance).mesh.set(rim, ball);
        const crown = block(world, [5, 1.75, -2], [0.15, 0.2, 0.15], [0.72, 0.4, 0.05, 1]);
        world.storage(MeshInstance).mesh.set(crown, ball);
        block(world, [5, 0.5, -2], [0.25, 1, 0.25], [0.658, 0.301, 0.117, 1]);
        const camera = world.create();
        world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        world.add(camera, Transform, { translation: [0, 1.7, 5] });
        world.add(camera, Listener);
        world.add(camera, InteractionInput);
        const player = world.create();
        world.add(player, Transform, { translation: [0, 1, 5] });
        world.add(player, Body, { type: BodyType.Kinematic });
        world.add(player, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.3, 0],
            capsuleCenter2: [0, 0.3, 0],
            radius: 0.3,
        });
        world.add(player, Character);
        world.add(player, Player, { camera });
        world.add(player, LocalPlayer);
        Object.assign(world.resource(Yard), { camera, player, speaker, bell });
    },
};
