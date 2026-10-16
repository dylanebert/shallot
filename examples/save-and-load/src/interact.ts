import {
    Body,
    BodyType,
    component,
    Devices,
    entity,
    InputPlugin,
    Outline,
    OutlinePlugin,
    type Plugin,
    Shape,
    type System,
    Transform,
    u8,
    type World,
} from "@dylanebert/shallot";
import { castRayClosest, type PhysicsQueryHit } from "@dylanebert/shallot/standard/physics";
import { LocalPlayer, Player, PlayerPlugin, UpdatePlayerControlSystem } from "./player";
import { encode, Kind, SAVE_KEY, Saved, Wallet } from "./save";

export const Action = { Coin: 0, Machine: 1, Crate: 2, Save: 3 } as const;
export const Usable = component("Usable", { action: u8 });
export const InteractionInput = component("InteractionInput", { target: entity });
export const InteractionPress = component("InteractionPress", { camera: entity, target: entity });
export const Hint = { create: () => ({ target: 0, action: "" }) };
export const Game = {
    create: () => ({
        player: 0,
        camera: 0,
        crate: createCrate,
        save: (_world: World): void => {},
    }),
};
export function createCrate(world: World): number {
    const eid = world.create();
    world.add(eid, Saved, { kind: Kind.Crate });
    world.add(eid, Transform, { translation: [-0.6, 0.35, -2.5], scale: [0.7, 0.7, 0.7] });
    world.add(eid, Usable, { action: Action.Crate });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    return eid;
}
const INPUTS = [InteractionInput, Transform];
const USABLES = [Usable];
const PRESSES = [InteractionPress];
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
        for (const camera of world.query(INPUTS)) {
            let yaw: number | undefined;
            let pitch = 0;
            for (const player of world.query(LOCAL_PLAYERS)) {
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
};

const UseInteraction: System = {
    name: "use",
    group: "fixed",
    update(world) {
        const game = world.resource(Game);
        const wallet = world.storage(Wallet);
        for (const press of world.query(PRESSES)) {
            const target = world.storage(InteractionPress).target.get(press);
            world.destroy(press);
            const carried = wallet.carried.get(game.player);
            if (
                target &&
                world.has(target, Usable) &&
                world.storage(Usable).action.get(target) === Action.Save
            ) {
                game.save(world);
            } else if (carried) {
                const t = world.storage(Transform);
                t.translation.y.set(carried, 0.35);
                t.rotation.set(carried, 0, 0, 0, 1);
                world.add(carried, Body, { type: BodyType.Static });
                world.add(carried, Shape);
                wallet.carried.set(game.player, 0);
            } else if (target && world.has(target, Usable)) {
                const money = wallet.money.get(game.player);
                switch (world.storage(Usable).action.get(target)) {
                    case Action.Coin:
                        wallet.money.set(game.player, money + 1);
                        world.destroy(target);
                        break;
                    case Action.Machine:
                        if (money < 3) break;
                        wallet.money.set(game.player, money - 3);
                        game.crate(world);
                        break;
                    case Action.Crate:
                        wallet.carried.set(game.player, target);
                        world.remove(target, Shape);
                        world.remove(target, Body);
                        break;
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
        const game = world.resource(Game);
        const carried = world.storage(Wallet).carried.get(game.player);
        if (carried) {
            const t = world.storage(Transform);
            const q = t.rotation;
            const x = q.x.get(game.camera),
                y = q.y.get(game.camera),
                z = q.z.get(game.camera),
                w = q.w.get(game.camera);
            const p = t.translation;
            p.set(
                carried,
                p.x.get(game.camera) - 2 * (x * z + w * y) * 1.2,
                p.y.get(game.camera) + 2 * (w * x - y * z) * 1.2 - 0.35,
                p.z.get(game.camera) - (1 - 2 * (x * x + y * y)) * 1.2,
            );
            q.set(carried, x, y, z, w);
        }
        const target = focus(world, game.camera);
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
        hint.target = target || carried;
        const action = target ? world.storage(Usable).action.get(target) : -1;
        hint.action =
            action === Action.Save
                ? "Save"
                : carried
                  ? "Set down"
                  : action === Action.Coin
                    ? "Pick up"
                    : action === Action.Machine
                      ? world.storage(Wallet).money.get(game.player) >= 3
                          ? "Buy crate · 3 coins"
                          : "Needs 3 coins"
                      : target
                        ? "Carry"
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
export function saveToBrowser(world: World): void {
    localStorage.setItem(SAVE_KEY, JSON.stringify(encode(world)));
}
