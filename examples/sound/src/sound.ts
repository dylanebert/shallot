import {
    AudioPlugin,
    component,
    Devices,
    entity,
    f32,
    InputPlugin,
    type Plugin,
    play,
    Sound,
    Time,
    Transform,
    u8,
    type World,
} from "@dylanebert/shallot";
import { castRayClosest, emptyQueryHit } from "@dylanebert/shallot/standard/physics";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const Usable = component("Usable", { bell: u8, clip: f32 });
export const Sliding = component("Sliding", { phase: f32 });
export const InteractionInput = component("InteractionInput", { target: entity });
export const InteractionPress = component("InteractionPress", { camera: entity, target: entity });
export const Hint = { create: () => ({ target: 0, action: "" }) };
const INPUTS = [InteractionInput, Transform];
const PLAYERS = [LocalPlayer, Player];
const PRESSES = [InteractionPress];
const SLIDERS = [Sliding, Transform];
export const CHIME_VOLUME = 0.35;
const REACH = 3;

const Aim = {
    create: () => ({
        hit: emptyQueryHit(),
        origin: { x: 0, y: 0, z: 0 },
        ray: { x: 0, y: 0, z: 0 },
    }),
};

export function focus(world: World, camera: number, yaw?: number, pitch = 0): number {
    const { hit, origin, ray } = world.resource(Aim);
    const transform = world.storage(Transform);
    const p = transform.translation;
    origin.x = p.x.get(camera);
    origin.y = p.y.get(camera);
    origin.z = p.z.get(camera);
    const q = transform.rotation;
    const x = q.x.get(camera),
        y = q.y.get(camera),
        z = q.z.get(camera),
        w = q.w.get(camera);
    ray.x = -2 * (x * z + w * y) * REACH;
    ray.y = -2 * (y * z - w * x) * REACH;
    ray.z = -(1 - 2 * (x * x + y * y)) * REACH;
    if (yaw !== undefined) {
        ray.x = -Math.cos(pitch) * Math.sin(yaw) * REACH;
        ray.y = Math.sin(pitch) * REACH;
        ray.z = -Math.cos(pitch) * Math.cos(yaw) * REACH;
    }
    return castRayClosest(world, origin, ray, hit) && world.has(hit.body, Usable) ? hit.body : 0;
}

export const SoundGamePlugin: Plugin = {
    name: "SoundGame",
    recovery: "stateless",
    dependencies: [AudioPlugin, InputPlugin],
    components: [Usable, Sliding, InteractionInput, InteractionPress],
    systems: [
        {
            name: "sample",
            group: "setup",
            after: PlayerPlugin.systems?.filter((system) => system.group === "setup"),
            update(world) {
                const devices = world.resource(Devices);
                for (const camera of world.query(INPUTS)) {
                    let yaw: number | undefined;
                    let pitch = 0;
                    for (const player of world.query(PLAYERS)) {
                        if (world.storage(Player).camera.get(player) !== camera) continue;
                        yaw = world.storage(LocalPlayer).viewYaw.get(player);
                        pitch = world.storage(LocalPlayer).viewPitch.get(player);
                    }
                    const target = focus(world, camera, yaw, pitch);
                    world.storage(InteractionInput).target.set(camera, target);
                    if (devices.enabled && !world.time.paused && devices.keys.pressed.has("KeyE")) {
                        const press = world.create();
                        world.add(press, InteractionPress, { camera, target });
                    }
                }
            },
        },
        {
            name: "use",
            group: "fixed",
            update(world) {
                const presses = world.storage(InteractionPress);
                for (const press of world.query(PRESSES)) {
                    const camera = presses.camera.get(press),
                        target = presses.target.get(press);
                    world.destroy(press);
                    if (!world.has(camera, InteractionInput) || !world.has(target, Usable))
                        continue;
                    if (world.storage(Usable).bell.get(target)) {
                        const p = world.storage(Transform).translation;
                        play(world, world.storage(Usable).clip.get(target), {
                            position: [p.x.get(target), p.y.get(target), p.z.get(target)],
                            volume: CHIME_VOLUME,
                        });
                    } else {
                        const paused = world.storage(Sound).paused;
                        paused.set(target, 1 - paused.get(target));
                    }
                }
                for (const eid of world.query(SLIDERS)) {
                    const phase = world.storage(Sliding).phase;
                    phase.set(eid, phase.get(eid) + Time.FIXED_DT * 0.3);
                    world.storage(Transform).translation.x.set(eid, Math.sin(phase.get(eid)) * 3);
                }
            },
        },
        {
            name: "hint",
            group: "simulation",
            after: [UpdatePlayerControlSystem],
            update(world) {
                const hint = world.resource(Hint);
                hint.target = 0;
                for (const camera of world.query(INPUTS)) hint.target = focus(world, camera);
                hint.action = !hint.target
                    ? ""
                    : world.storage(Usable).bell.get(hint.target)
                      ? "Ring bell"
                      : world.storage(Sound).paused.get(hint.target)
                        ? "Resume music"
                        : "Pause music";
            },
        },
    ],
};
