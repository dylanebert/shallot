import {
    Body,
    BodyType,
    Camera,
    Character,
    GlobalAmbientLight,
    internText,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    PointLight,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    sphere,
    Text,
    TextPlugin,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Billboard } from "@dylanebert/shallot/rendering";
import {
    Action,
    createCrate,
    Game,
    InteractionInput,
    InteractPlugin,
    saveToBrowser,
    Usable,
} from "./interact";
import { LocalPlayer, Player, PlayerPlugin } from "./player";
import { decode, Kind, readSave, type Save, Saved, Wallet } from "./save";

const GROUND = [0.085, 0.08, 0.072, 1] as const;
const AMBER = [0.72, 0.4, 0.05, 1] as const;
const CYAN = [0.04, 0.43, 0.53, 1] as const;
const ORANGE = [0.658, 0.301, 0.117, 1] as const;
const CoinMesh = {
    create: (world: World) => world.resource(Meshes).add({ name: "coin", ...sphere() }),
};
type Vec3 = readonly [number, number, number];
type Color = readonly [number, number, number, number];
function mesh(world: World, eid: number, color: Color): void {
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: color })),
    );
}
function block(world: World, position: Vec3, size: Vec3, color: Color = GROUND): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale: size });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    mesh(world, eid, color);
    return eid;
}
function saved(world: World, kind: number, translation: Vec3, scale: Vec3 = [1, 1, 1]): number {
    const eid = world.create();
    world.add(eid, Saved, { kind });
    world.add(eid, Transform, { translation, scale });
    return eid;
}

export function newGame(world: World): number[] {
    const camera = saved(world, Kind.Camera, [0, 1.7, 2]);
    const player = saved(world, Kind.Player, [0, 1, 2]);
    world.add(player, Player, { camera });
    world.add(player, Wallet);
    const coins = ([-1.5, 0, 1.5] as const).map((x) =>
        saved(world, Kind.Coin, [x, 1.1, 0], [0.35, 0.35, 0.08]),
    );
    return [camera, player, ...coins];
}

export function populate(world: World, save: Save | null): number[] {
    return save ? decode(world, save) : newGame(world);
}

function decorate(world: World, eid: number): void {
    const game = world.resource(Game);
    switch (world.storage(Saved).kind.get(eid)) {
        case Kind.Camera:
            world.add(eid, Camera, { fov: 70, clearColor: 0x191816 });
            world.add(eid, StandardRenderer);
            world.add(eid, InteractionInput);
            break;
        case Kind.Player:
            world.add(eid, Body, { type: BodyType.Kinematic });
            world.add(eid, Shape, {
                kind: ShapeKind.Capsule,
                capsuleCenter1: [0, -0.3, 0],
                capsuleCenter2: [0, 0.3, 0],
                radius: 0.3,
            });
            world.add(eid, Character);
            world.add(eid, LocalPlayer);
            break;
        default: {
            const coin = world.storage(Saved).kind.get(eid) === Kind.Coin;
            mesh(world, eid, coin ? AMBER : ORANGE);
            if (coin) world.storage(MeshInstance).mesh.set(eid, world.resource(CoinMesh));
            world.add(eid, Usable, { action: coin ? Action.Coin : Action.Crate });
            if (world.storage(Wallet).carried.get(game.player) !== eid) {
                world.add(eid, Body, { type: BodyType.Static });
                world.add(eid, Shape);
            }
        }
    }
}

function room(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 300;
    const light = world.create();
    world.add(light, Transform, { translation: [0, 2.9, 0] });
    world.add(light, PointLight, {
        color: 0xffe8cc,
        intensity: 80000,
        range: 10,
        shadowMapsEnabled: 1,
    });
    const surfaces = [
        [
            [0, -0.25, 0],
            [8, 0.5, 8],
        ],
        [
            [0, 3.45, 0],
            [8, 0.5, 8],
        ],
        [
            [0, 1.6, -4],
            [8, 3.2, 0.4],
        ],
        [
            [0, 1.6, 4],
            [8, 3.2, 0.4],
        ],
        [
            [-4, 1.6, 0],
            [0.4, 3.2, 8],
        ],
        [
            [4, 1.6, 0],
            [0.4, 3.2, 8],
        ],
    ] as const;
    for (const [position, size] of surfaces) block(world, position, size);
    for (const x of [-1.5, 0, 1.5]) block(world, [x, 0.4, 0], [0.65, 0.8, 0.65]);
    const machine = block(world, [-1.8, 0.85, -2.5], [1, 1.7, 0.8], CYAN);
    const savePoint = block(world, [1.8, 0.85, -2.5], [0.6, 1.7, 0.6], AMBER);
    world.add(machine, Usable, { action: Action.Machine });
    world.add(savePoint, Usable, { action: Action.Save });
    for (const [x, label] of [
        [-1.8, "CRATES · 3"],
        [1.8, "SAVE"],
    ] as const) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 2.1, -2.5] });
        world.add(eid, Text, {
            content: internText(world, label),
            fontSize: 0.2,
            anchor: [0.5, 0],
            billboard: Billboard.Screen,
            color: 0xeee9df,
        });
    }
    const eids = populate(world, readSave(localStorage));
    const game = world.resource(Game);
    game.player = eids.find((eid) => world.storage(Saved).kind.get(eid) === Kind.Player)!;
    game.camera = world.storage(Player).camera.get(game.player);
    for (const eid of eids) decorate(world, eid);
    game.crate = (w) => {
        const eid = createCrate(w);
        mesh(w, eid, ORANGE);
        return eid;
    };
    game.save = saveToBrowser;
}
export const RoomPlugin: Plugin = {
    name: "Room",
    components: [Saved, Wallet],
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, InteractPlugin, TextPlugin],
    initialize: room,
};
