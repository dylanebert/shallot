import {
    AudioOutput,
    Body,
    BodyType,
    Camera,
    Character,
    DirectionalLight,
    GlobalAmbientLight,
    Listener,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    registerClip,
    Shape,
    ShapeKind,
    Sound,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { loadClip, StandardAudioPlugin } from "@dylanebert/shallot/standard/audio";
import { chime } from "./chime";
import { LocalPlayer, Player, PlayerPlugin } from "./player";
import { InteractionInput, Sliding, SoundGamePlugin, Usable } from "./sound";

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
        const sampleRate = world.resource(AudioOutput).context?.sampleRate ?? 48000;
        const tone = registerClip(world, chime(sampleRate), sampleRate);
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = 300;
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 6000,
            shadowMapsEnabled: 1,
        });
        block(world, [0, -0.25, 0], [40, 0.5, 40], [0.085, 0.08, 0.072, 1]);
        for (const x of [-12, 12])
            block(world, [x, 0.5, 0], [0.3, 1, 24], [0.042, 0.039, 0.035, 1]);
        for (const z of [-12, 12])
            block(world, [0, 0.5, z], [24, 1, 0.3], [0.042, 0.039, 0.035, 1]);
        const speaker = block(world, [0, 1.4, -2], [1, 1, 1], [0.04, 0.43, 0.53, 1]);
        world.storage(Body).type.set(speaker, BodyType.Kinematic);
        world.add(speaker, Sound, { clip: music, loop: 1, volume: 0.7 });
        world.add(speaker, Sliding);
        world.add(speaker, Usable);
        const bell = block(world, [5, 1.4, -2], [1, 1, 1], [0.72, 0.4, 0.05, 1]);
        world.add(bell, Usable, { bell: 1, clip: tone });
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
