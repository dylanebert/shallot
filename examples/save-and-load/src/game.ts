import {
    component,
    Devices,
    InputPlugin,
    internText,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    Outline,
    type Plugin,
    StandardMaterial,
    type System,
    Text,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { viewportToWorld } from "@dylanebert/shallot/rendering";
import { castRayClosest, emptyQueryHit } from "@dylanebert/shallot/standard/physics";
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";
import {
    COIN_CAP,
    COIN_DIAMETER,
    COIN_THICKNESS,
    Coin,
    decode,
    encode,
    placement,
    readSave,
    SAVE_KEY,
} from "./save";

export const Cube = component("Cube", {});
export const Game = {
    create: () => ({
        camera: 0,
        cube: 0,
        label: 0,
        press: null as { x: number; y: number } | null,
        storage: null as Pick<Storage, "getItem" | "setItem"> | null,
        saved: false,
        feedback: 0,
        mesh: 0,
        material: null as MaterialHandle | null,
        hit: emptyQueryHit(),
    }),
};

export function count(world: World): number {
    let total = 0;
    for (const _eid of world.query([Coin])) total++;
    return total;
}
export function newGame(world: World): void {
    for (const eid of world.query([Coin])) world.destroy(eid);
    world.resource(Game).feedback = 0;
}
export function saveGame(world: World): void {
    const game = world.resource(Game);
    if (!game.storage) return;
    game.storage.setItem(SAVE_KEY, JSON.stringify(encode(world)));
    game.saved = true;
    game.feedback = 2;
}
export function loadGame(world: World): void {
    const game = world.resource(Game);
    const save = game.storage && readSave(game.storage);
    if (!save) return;
    for (const eid of decode(world, save)) presentCoin(world, eid);
    game.feedback = 0;
}
function presentCoin(world: World, eid: number): void {
    const game = world.resource(Game);
    world.add(eid, MeshInstance, { mesh: game.mesh });
    if (game.material) world.add(eid, MeshMaterial, game.material);
}
export function saveLabel(world: World): string {
    return world.resource(Game).feedback > 0 ? "Saved" : "Save";
}
export function addCoin(world: World): number | null {
    const total = count(world);
    if (total >= COIN_CAP) return null;
    const position = placement(total);
    const eid = world.create();
    world.add(eid, Coin, { height: position[1], speed: 0 });
    world.add(eid, Transform, {
        translation: [position[0], position[1] + 1.2, position[2]],
        scale: [COIN_DIAMETER, COIN_THICKNESS, COIN_DIAMETER],
    });
    presentCoin(world, eid);
    return eid;
}
export function hitsCube(world: World, x: number, y: number): boolean {
    const game = world.resource(Game);
    const ray = viewportToWorld(world, game.camera, x, y);
    if (!ray) return false;
    return (
        castRayClosest(
            world,
            { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
            { x: ray.dir[0] * 100, y: ray.dir[1] * 100, z: ray.dir[2] * 100 },
            game.hit,
        ) && game.hit.body === game.cube
    );
}
export const Play: System = {
    name: "Game/Play",
    group: "simulation",
    update(world) {
        const game = world.resource(Game);
        const input = world.resource(Devices);
        const pointer = input.pointer;
        game.feedback = Math.max(0, game.feedback - world.time.deltaTime);
        if (input.enabled) {
            if (input.keys.pressed.has("KeyN")) newGame(world);
            if (input.keys.pressed.has("KeyS")) saveGame(world);
            if (input.keys.pressed.has("KeyL")) loadGame(world);
            for (const event of pointer.buttonEvents) {
                if (event.button !== "left") continue;
                if (event.pressed)
                    game.press = hitsCube(world, event.x, event.y)
                        ? { x: event.x, y: event.y }
                        : null;
                else {
                    if (
                        game.press &&
                        Math.hypot(event.x - game.press.x, event.y - game.press.y) <= 5 &&
                        hitsCube(world, event.x, event.y)
                    )
                        addCoin(world);
                    game.press = null;
                }
            }
            if (
                game.press &&
                (!pointer.left ||
                    Math.hypot(pointer.x - game.press.x, pointer.y - game.press.y) > 5)
            )
                game.press = null;
        } else game.press = null;
        if (game.label)
            world
                .storage(Text)
                .content.set(
                    game.label,
                    internText(
                        world,
                        count(world) >= COIN_CAP ? "The pile is full" : "Click for a coin",
                    ),
                );
        if (input.enabled && pointer.hover && hitsCube(world, pointer.x, pointer.y)) {
            if (!world.has(game.cube, Outline))
                world.add(game.cube, Outline, {
                    color: 0xeee9df,
                    width: 4,
                });
        } else if (world.has(game.cube, Outline)) world.remove(game.cube, Outline);
    },
};
export const Fall: System = {
    name: "Game/Fall",
    group: "simulation",
    update(world) {
        const coin = world.storage(Coin);
        const y = world.storage(Transform).translation.y;
        for (const eid of world.query([Coin])) {
            const height = coin.height.get(eid);
            if (y.get(eid) <= height) continue;
            const speed = coin.speed.get(eid) + 8 * world.time.deltaTime;
            coin.speed.set(eid, speed);
            y.set(eid, Math.max(height, y.get(eid) - speed * world.time.deltaTime));
        }
    },
};

export function coinAssets(world: World): void {
    // A flat rim and separate cap normals keep the little discs readable at orbit distance.
    const vertices: number[] = [],
        indices: number[] = [];
    for (let side = 0; side < 24; side++) {
        const a = (side * Math.PI) / 12,
            b = ((side + 1) * Math.PI) / 12;
        for (const top of [-1, 1]) {
            const start = vertices.length / 8;
            for (const [x, z] of [
                [0, 0],
                [Math.cos(a) * 0.5, Math.sin(a) * 0.5],
                [Math.cos(b) * 0.5, Math.sin(b) * 0.5],
            ])
                vertices.push(x!, top * 0.5, z!, 0, 0, top, 0, 0);
            indices.push(start, start + (top === 1 ? 2 : 1), start + (top === 1 ? 1 : 2));
        }
        const start = vertices.length / 8;
        for (const [angle, y] of [
            [a, -0.5],
            [a, 0.5],
            [b, 0.5],
            [b, -0.5],
        ])
            vertices.push(
                Math.cos(angle!) * 0.5,
                y!,
                Math.sin(angle!) * 0.5,
                0,
                Math.cos(angle!),
                0,
                Math.sin(angle!),
                0,
            );
        indices.push(start, start + 1, start + 2, start, start + 2, start + 3);
    }
    const game = world.resource(Game);
    game.mesh = world.resource(Meshes).add({
        name: "coin",
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
    });
    game.material = world.resource(Materials).add(
        StandardMaterial({
            baseColor: [0.55, 0.29, 0.035, 1],
            metallic: 0.65,
            perceptualRoughness: 0.3,
        }),
    );
}
export const GamePlugin: Plugin = {
    name: "Game",
    components: [Cube, Coin],
    dependencies: [InputPlugin],
    systems: [Play, Fall],
};
