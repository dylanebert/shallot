import {
    component,
    Devices,
    entity,
    InputPlugin,
    internText,
    NotShadowCaster,
    Outline,
    OutlinePlugin,
    type Plugin,
    PointLight,
    resetInterpolation,
    Sprite,
    SpriteFill,
    SpritePlugin,
    type System,
    Text,
    TextPlugin,
    Time,
    Transform,
    u8,
    u32,
    type World,
} from "@dylanebert/shallot";
import { Billboard } from "@dylanebert/shallot/rendering";
import { castRayClosest, type PhysicsQueryHit } from "@dylanebert/shallot/standard/physics";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const Action = { Lamp: 0, Door: 1, Button: 2, Hold: 3 } as const;
export const Usable = component("Usable", {
    action: u8,
    active: u8,
    light: entity,
    label: u32,
    activeLabel: u32,
});
/** Sampled E input and its fixed-tick hold state, all captured with the world. */
export const InteractionInput = component("InteractionInput", {
    target: entity,
    presses: u32,
    held: u8,
    /** Release or lost focus stays latched until a tick consumes it. */
    cancelled: u8,
    lastPresses: u32,
    holding: entity,
    ticks: u32,
    spent: u8,
});
export const Prompt = component("Prompt", { key: entity, label: entity, ring: entity });
const INPUTS = [InteractionInput, Transform];
const USABLES = [Usable];
const PROMPTS = [Prompt];
const LOCAL_PLAYERS = [LocalPlayer, Player];
const REACH = 3;
const Aim = {
    create: () => ({
        hit: {
            shape: 0,
            body: 0,
            point: { x: 0, y: 0, z: 0 },
            normal: { x: 0, y: 0, z: 0 },
            fraction: 0,
            userMaterialId: 0n,
            triangleIndex: 0,
            childIndex: 0,
        } satisfies PhysicsQueryHit,
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
            if (target !== input.target.get(camera) || !held) input.cancelled.set(camera, 1);
            input.target.set(camera, target);
            input.held.set(camera, held);
            if (devices.enabled && !world.time.paused && devices.keys.pressed.has("KeyE"))
                input.presses.set(camera, input.presses.get(camera) + 1);
        }
    },
};

function activate(world: World, target: number): void {
    const usable = world.storage(Usable);
    const active = 1 - usable.active.get(target);
    usable.active.set(target, active);
    switch (usable.action.get(target)) {
        case Action.Lamp:
            world.storage(PointLight).intensity.set(usable.light.get(target), active ? 1800 : 0);
            break;
        case Action.Door:
            world.storage(Transform).translation.y.set(target, active ? 4 : 1.4);
            break;
        case Action.Button:
            world.storage(Transform).translation.z.set(target, active ? -3.92 : -3.8);
            break;
        case Action.Hold:
            world.storage(Transform).translation.y.set(target, active ? 2.2 : 1.2);
            break;
    }
}

const UseInteraction: System = {
    name: "use",
    group: "fixed",
    update(world) {
        const input = world.storage(InteractionInput);
        for (const camera of world.query(INPUTS)) {
            const target = input.target.get(camera);
            const pressed = input.presses.get(camera) !== input.lastPresses.get(camera);
            input.lastPresses.set(camera, input.presses.get(camera));
            if (input.cancelled.get(camera)) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 1);
                input.cancelled.set(camera, 0);
            }
            if (pressed) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 0);
                input.holding.set(camera, target);
            }
            if (!target || !world.has(target, Usable)) {
                input.ticks.set(camera, 0);
                input.spent.set(camera, 1);
                continue;
            }
            if (world.storage(Usable).action.get(target) !== Action.Hold) {
                if (pressed) activate(world, target);
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
        let camera = 0;
        for (const eid of world.query(INPUTS)) {
            camera = eid;
            target = focus(world, eid);
            ticks = input.ticks.get(eid);
        }
        for (const eid of world.query(USABLES)) {
            if (eid === target) {
                if (!world.has(eid, Outline))
                    world.add(eid, Outline, {
                        color: [0.8655607, 0.76723723, 0.62240485, 1],
                        width: 4,
                    });
            } else if (world.has(eid, Outline)) world.remove(eid, Outline);
        }
        const prompts = world.storage(Prompt);
        for (const eid of world.query(PROMPTS)) {
            const key = prompts.key.get(eid),
                label = prompts.label.get(eid),
                ring = prompts.ring.get(eid);
            world.storage(Text).visible.set(key, Number(!!target));
            world.storage(Text).visible.set(label, Number(!!target));
            world.storage(Sprite).visible.set(eid, Number(!!target));
            world
                .storage(Sprite)
                .visible.set(ring, Number(!!target && usable.action.get(target) === Action.Hold));
            if (!target) continue;
            const active = usable.active.get(target);
            world
                .storage(Text)
                .content.set(
                    label,
                    active ? usable.activeLabel.get(target) : usable.label.get(target),
                );
            world.storage(Sprite).fill.set(ring, ticks / 60);
            const t = world.storage(Transform);
            const x = t.translation.x.get(target) + t.scale.x.get(target) / 2 + 0.23,
                y = t.translation.y.get(target),
                z = t.translation.z.get(target) + t.scale.z.get(target) / 2 + 0.08;
            const q = t.rotation;
            const rightX = 1 - 2 * (q.y.get(camera) ** 2 + q.z.get(camera) ** 2);
            const rightY =
                2 * (q.x.get(camera) * q.y.get(camera) + q.w.get(camera) * q.z.get(camera));
            const rightZ =
                2 * (q.x.get(camera) * q.z.get(camera) - q.w.get(camera) * q.y.get(camera));
            t.translation.set(eid, x, y, z);
            t.translation.set(key, x, y, z);
            t.translation.set(ring, x, y, z);
            t.translation.set(label, x + rightX * 0.2, y + rightY * 0.2, z + rightZ * 0.2);
            // A focus change is immediate UI placement, not motion between fixed poses.
            resetInterpolation(world, eid);
            resetInterpolation(world, key);
            resetInterpolation(world, ring);
            resetInterpolation(world, label);
        }
    },
};

export const InteractPlugin: Plugin = {
    name: "Interact",
    recovery: "stateless",
    components: [Usable, InteractionInput, Prompt],
    dependencies: [InputPlugin, OutlinePlugin, SpritePlugin, TextPlugin],
    systems: [SampleInteraction, UseInteraction, ShowInteraction],
};

export function makePrompt(world: World, image: number, ringImage: number): number {
    const cap = world.create();
    world.add(cap, Transform);
    world.add(cap, NotShadowCaster);
    world.add(cap, Sprite, { image, size: [0.28, 0.28], billboard: Billboard.Screen });
    const key = world.create();
    world.add(key, Transform);
    world.add(key, Text, {
        content: internText(world, "E"),
        fontSize: 0.18,
        anchor: [0.5, 0.5],
        billboard: Billboard.Screen,
        color: 0xefe3cf,
    });
    const label = world.create();
    world.add(label, Transform);
    world.add(label, Text, {
        fontSize: 0.16,
        anchor: [0, 0.5],
        billboard: Billboard.Screen,
        color: 0xefe3cf,
    });
    const ring = world.create();
    world.add(ring, Transform);
    world.add(ring, NotShadowCaster);
    world.add(ring, Sprite, {
        image: ringImage,
        size: [0.36, 0.36],
        billboard: Billboard.Screen,
        fillMode: SpriteFill.Radial,
        fill: 0,
    });
    world.add(cap, Prompt, { key, label, ring });
    return cap;
}
