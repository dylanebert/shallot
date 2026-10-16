import {
    Content,
    component,
    Devices,
    entity,
    InputPlugin,
    Materials,
    MeshMaterial,
    Outline,
    OutlinePlugin,
    type Plugin,
    PointLight,
    type System,
    Transform,
    u8,
    u32,
    type World,
} from "@dylanebert/shallot";
import { castRayClosest, type PhysicsQueryHit } from "@dylanebert/shallot/standard/physics";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";

export const LAMP_EMISSION = [2.88, 1.6, 0.2] as const;

export const Usable = component("Usable", {
    lamp: entity,
    active: u8,
    label: u32,
    activeLabel: u32,
});
export const InteractionInput = component("InteractionInput", { target: entity });
export const InteractionPress = component("InteractionPress", { target: entity });
const PRESSES = [InteractionPress];
export const Hint = { create: () => ({ target: 0, action: "", hold: false, progress: 0 }) };
const INPUTS = [InteractionInput, Transform];
const USABLES = [Usable];
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
            input.target.set(camera, target);
            if (devices.enabled && !world.time.paused && devices.keys.pressed.has("KeyE")) {
                const press = world.create();
                world.add(press, InteractionPress, { target });
            }
        }
    },
};

const UseInteraction: System = {
    name: "use",
    group: "fixed",
    update(world) {
        const usable = world.storage(Usable);
        for (const press of world.query(PRESSES)) {
            const target = world.storage(InteractionPress).target.get(press);
            world.destroy(press);
            if (!target || !world.has(target, Usable)) continue;
            const active = 1 - usable.active.get(target);
            usable.active.set(target, active);
            const lamp = usable.lamp.get(target);
            world.storage(PointLight).intensity.set(lamp, active ? 80000 : 0);
            if (world.has(lamp, MeshMaterial))
                world.resource(Materials).update(world.storage(MeshMaterial).material.get(lamp), {
                    emissive: active ? LAMP_EMISSION : [0, 0, 0],
                });
        }
    },
};

const ShowInteraction: System = {
    name: "prompt",
    group: "simulation",
    after: [UpdatePlayerControlSystem],
    update(world) {
        const usable = world.storage(Usable);
        let target = 0;
        for (const eid of world.query(INPUTS)) target = focus(world, eid);
        for (const eid of world.query(USABLES)) {
            if (eid === target) {
                if (!world.has(eid, Outline))
                    world.add(eid, Outline, {
                        color: [0.8549926, 0.8148466, 0.7379104, 1],
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
    },
};

export const InteractPlugin: Plugin = {
    name: "Interact",
    recovery: "stateless",
    components: [Usable, InteractionInput, InteractionPress],
    dependencies: [InputPlugin, OutlinePlugin],
    systems: [SampleInteraction, UseInteraction, ShowInteraction],
};
