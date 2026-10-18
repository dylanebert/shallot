import {
    Content,
    component,
    Devices,
    entity,
    f32,
    InputPlugin,
    Materials,
    MeshMaterial,
    Outline,
    OutlinePlugin,
    type Plugin,
    PointLight,
    type System,
    TextPlugin,
    Time,
    Transform,
    u8,
    u32,
    type World,
} from "@dylanebert/shallot";
import { castRayClosest, emptyQueryHit } from "@dylanebert/shallot/standard/physics";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const Action = { Lamp: 0, Button: 1, Hold: 2 } as const;
export const Usable = component("Usable", {
    action: u8,
    active: u8,
    label: u32,
    activeLabel: u32,
    presses: u32,
    remaining: u32,
    rest: f32,
});
/** Current aim, sampled held state and fixed-tick hold progress. */
export const InteractionInput = component("InteractionInput", {
    target: entity,
    held: u8,
    /** Release or lost focus stays latched until a tick consumes it. */
    cancelled: u8,
    holding: entity,
    ticks: u32,
    spent: u8,
});
/** One sampled press until its consuming tick; a zero target invalidates a pending hold start. */
export const InteractionPress = component("InteractionPress", { camera: entity, target: entity });
const PRESSES = [InteractionPress];
export const Hint = { create: () => ({ target: 0, action: "", hold: false, progress: 0 }) };
const INPUTS = [InteractionInput, Transform];
const USABLES = [Usable];
const LOCAL_PLAYERS = [LocalPlayer, Player];
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

const SampleInteraction: System = {
    name: "sample",
    group: "setup",
    after: PlayerPlugin.systems?.filter((system) => system.group === "setup"),
    update(world) {
        const devices = world.resource(Devices);
        const input = world.storage(InteractionInput);
        for (const camera of world.query(INPUTS)) {
            let yaw: number | undefined;
            let pitch = 0;
            for (const player of world.query(LOCAL_PLAYERS)) {
                if (world.storage(Player).camera.get(player) !== camera) continue;
                yaw = world.storage(LocalPlayer).viewYaw.get(player);
                pitch = world.storage(LocalPlayer).viewPitch.get(player);
            }
            const target = focus(world, camera, yaw, pitch);
            const held = Number(devices.enabled && devices.keys.held.has("KeyE"));
            if (target !== input.target.get(camera) || !held) {
                input.cancelled.set(camera, 1);
                const presses = world.storage(InteractionPress);
                for (const press of world.query(PRESSES)) {
                    const pending = presses.target.get(press);
                    if (
                        presses.camera.get(press) === camera &&
                        world.has(pending, Usable) &&
                        world.storage(Usable).action.get(pending) === Action.Hold
                    )
                        presses.target.set(press, 0);
                }
            }
            input.target.set(camera, target);
            input.held.set(camera, held);
            if (devices.enabled && !world.time.paused && devices.keys.pressed.has("KeyE")) {
                const press = world.create();
                world.add(press, InteractionPress, { camera, target });
            }
        }
    },
};

function activate(world: World, target: number): void {
    const usable = world.storage(Usable);
    const active = 1 - usable.active.get(target);
    usable.presses.set(target, usable.presses.get(target) + 1);
    switch (usable.action.get(target)) {
        case Action.Lamp:
            usable.active.set(target, active);
            world.storage(PointLight).intensity.set(target, active ? 150000 : 0);
            if (world.has(target, MeshMaterial))
                world.resource(Materials).update(world.storage(MeshMaterial).material.get(target), {
                    emissive: active ? [2.88, 1.6, 0.2] : [0, 0, 0],
                });
            break;
        case Action.Button: {
            const z = world.storage(Transform).translation.z;
            if (!usable.remaining.get(target)) usable.rest.set(target, z.get(target));
            usable.remaining.set(target, 11);
            z.set(target, usable.rest.get(target) - 0.025 * Math.sin(Math.PI / 12));
            break;
        }
        case Action.Hold: {
            usable.active.set(target, active);
            if (world.has(target, MeshMaterial)) {
                const material = world.storage(MeshMaterial);
                world.resource(Materials).update(material.material.get(target), {
                    baseColor: active ? [0.72, 0.4, 0.05, 1] : [0.658, 0.301, 0.117, 1],
                });
            }
            break;
        }
    }
}

const UseInteraction: System = {
    name: "use",
    group: "fixed",
    update(world) {
        const input = world.storage(InteractionInput);
        const usable = world.storage(Usable);
        for (const target of world.query(USABLES)) {
            const remaining = usable.remaining.get(target);
            if (!remaining) continue;
            usable.remaining.set(target, remaining - 1);
            world
                .storage(Transform)
                .translation.z.set(
                    target,
                    usable.rest.get(target) -
                        (remaining === 1 ? 0 : 0.025 * Math.sin((Math.PI * (13 - remaining)) / 12)),
                );
        }
        for (const camera of world.query(INPUTS)) {
            if (input.cancelled.get(camera)) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 1);
                input.cancelled.set(camera, 0);
            }
        }
        const presses = world.storage(InteractionPress);
        for (const press of world.query(PRESSES)) {
            const camera = presses.camera.get(press),
                target = presses.target.get(press);
            world.destroy(press);
            if (!world.has(camera, InteractionInput) || !target || !world.has(target, Usable))
                continue;
            if (usable.action.get(target) !== Action.Hold) activate(world, target);
            else if (input.held.get(camera) && input.target.get(camera) === target) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 0);
                input.holding.set(camera, target);
            }
        }
        for (const camera of world.query(INPUTS)) {
            const target = input.target.get(camera);
            if (!target || !world.has(target, Usable)) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 1);
                continue;
            }
            if (world.storage(Usable).action.get(target) !== Action.Hold) {
                input.ticks.set(camera, 0);
                continue;
            }
            if (!input.held.get(camera) || input.holding.get(camera) !== target) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 1);
            } else if (!input.spent.get(camera)) {
                const ticks = input.ticks.get(camera) + 1;
                input.ticks.set(camera, ticks);
                if (ticks === Math.round(1 / Time.FIXED_DT)) {
                    activate(world, target);
                    input.spent.set(camera, 1);
                }
            }
        }
    },
};

const ShowInteraction: System = {
    name: "prompt",
    group: "simulation",
    after: [UpdatePlayerControlSystem],
    update(world) {
        const usable = world.storage(Usable);
        const input = world.storage(InteractionInput);
        let target = 0;
        let ticks = 0;
        for (const eid of world.query(INPUTS)) {
            target = focus(world, eid);
            ticks = input.ticks.get(eid);
        }
        for (const eid of world.query(USABLES)) {
            if (eid === target) {
                if (!world.has(eid, Outline))
                    world.add(eid, Outline, {
                        color: 0xeee9df,
                        width: 4,
                    });
            } else if (world.has(eid, Outline)) world.remove(eid, Outline);
        }
        const hint = world.resource(Hint);
        hint.target = target;
        hint.action = target
            ? world
                  .resource(Content)
                  .name(
                      usable.active.get(target)
                          ? usable.activeLabel.get(target)
                          : usable.label.get(target),
                  )!
            : "";
        hint.hold = !!target && usable.action.get(target) === Action.Hold;
        hint.progress = ticks / 60;
    },
};

export const InteractPlugin: Plugin = {
    name: "Interact",
    recovery: "stateless",
    components: [Usable, InteractionInput, InteractionPress],
    dependencies: [InputPlugin, OutlinePlugin, TextPlugin],
    systems: [SampleInteraction, UseInteraction, ShowInteraction],
};
