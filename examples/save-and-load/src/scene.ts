import {
    Body,
    BodyType,
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    internText,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    OutlinePlugin,
    type Plugin,
    Shape,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Text,
    TextPlugin,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { Billboard } from "@dylanebert/shallot/rendering";
import { Cube, coinAssets, Game, GamePlugin } from "./game";
import { readSave } from "./save";

function makeScene(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 380;

    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);

    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 6, yaw: 0.7, pitch: 0.4 });
    world.add(camera, Transform);

    const materials = world.resource(Materials);
    const cubeMaterial = materials.add(
        StandardMaterial({
            baseColor: [0.72, 0.4, 0.05, 1],
            perceptualRoughness: 0.38,
        }),
    );
    const cube = world.create();
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, cubeMaterial);
    world.add(cube, Transform, { translation: [0, 0, 0] });
    world.add(cube, Cube);
    world.add(cube, Body, { type: BodyType.Static });
    world.add(cube, Shape);

    const groundMaterial = materials.add(
        StandardMaterial({
            baseColor: [0.085, 0.08, 0.072, 1],
            perceptualRoughness: 0.82,
        }),
    );
    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(ground, MeshMaterial, groundMaterial);
    world.add(ground, Transform, {
        translation: [0, -0.6, 0],
        scale: [4, 0.2, 4],
    });
    const label = world.create();
    world.add(label, Transform, { translation: [0, 1, 0] });
    world.add(label, Text, {
        content: internText(world, "Click for a coin"),
        fontSize: 0.18,
        anchor: [0.5, 0],
        billboard: Billboard.Screen,
        color: 0xeee9df,
    });
    const game = world.resource(Game);
    Object.assign(game, { camera, cube });
    if (typeof localStorage !== "undefined") {
        game.storage = localStorage;
        game.saved = readSave(localStorage) !== null;
    }
    coinAssets(world);
}

export const Scene: Plugin = {
    name: "Scene",
    dependencies: [OrbitPlugin, StandardPhysicsPlugin, OutlinePlugin, TextPlugin, GamePlugin],
    initialize: makeScene,
};
