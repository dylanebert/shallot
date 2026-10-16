import {
    Body,
    BodyType,
    Camera,
    CameraBackground,
    Character,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    PointLight,
    type Resource,
    Shape,
    ShapeKind,
    SpotLight,
    StandardMaterial,
    StandardPhysicsPlugin,
    StandardRenderer,
    Transform,
    VolumetricLight,
    type World,
} from "@dylanebert/shallot";
import { Fog, FogPlugin, Sky, SkyPlugin, skyBackgroundHandle } from "@dylanebert/shallot/extras";
import { LocalPlayer, Player, PlayerPlugin } from "./player";

type Color = readonly [number, number, number, number];
type Vec3 = readonly [number, number, number];
const CARBON: Color = [0.085, 0.08, 0.072, 1];
const TRIM: Color = [0.042, 0.039, 0.035, 1];
const AMBER: Color = [0.72, 0.4, 0.05, 1];
function block(world: World, position: Vec3, size: Vec3, color: Color = CARBON): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale: size });
    world.add(eid, Body, { type: BodyType.Static });
    world.add(eid, Shape);
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: color, perceptualRoughness: 0.85 })),
    );
    return eid;
}
export const Room: Resource<{
    camera: number;
    player: number;
    sun: number;
    fog: number;
    pendants: number[];
    downlight: number;
    object: number;
    bulbs: number[];
}> = {
    create: () => ({
        camera: 0,
        player: 0,
        sun: 0,
        fog: 0,
        pendants: [],
        downlight: 0,
        object: 0,
        bulbs: [],
    }),
};
function room(world: World): void {
    const scene = world.resource(Room);
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 180;
    block(world, [0, -0.25, 0], [9, 0.5, 24]);
    block(world, [0, 5.25, 0], [9, 0.5, 24], [0.11, 0.105, 0.095, 1]);
    for (const [position, size] of [
        [
            [-4.5, 2.5, 0],
            [0.4, 5, 24],
        ],
        [
            [4.5, 2.5, 0],
            [0.4, 5, 24],
        ],
        [
            [0, 2.5, 12],
            [9, 5, 0.4],
        ],
        [
            [-3.25, 2.5, -12],
            [2.5, 5, 0.4],
        ],
        [
            [3.25, 2.5, -12],
            [2.5, 5, 0.4],
        ],
        [
            [0, 0.75, -12],
            [4, 1.5, 0.4],
        ],
        [
            [0, 4.75, -12],
            [4, 0.5, 0.4],
        ],
    ] as const)
        block(world, position, size, [0.15, 0.14, 0.125, 1]);
    scene.sun = world.create();
    world.add(scene.sun, Transform, { rotation: lookAtRotation(0, 8, -12, 0, 0, 0) });
    world.add(scene.sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 70000,
        shadowMapsEnabled: 1,
        maximumDistance: 45,
    });
    world.add(scene.sun, VolumetricLight);
    scene.fog = world.create();
    world.add(scene.fog, Fog, {
        color: 0,
        density: 0.035,
        heightFalloff: 0,
        scatteringAsymmetry: 0.25,
        lightIntensity: 0.5 / Math.PI,
        stepCount: 64,
        jitter: 0,
    });
    for (const z of [4, 8]) {
        block(world, [-1, 4.4, z], [0.04, 1.2, 0.04], TRIM);
        block(world, [-1, 3.98, z], [0.65, 0.12, 0.65], TRIM);
        const bulb = block(world, [-1, 3.64, z], [0.18, 0.16, 0.18], AMBER);
        world.add(bulb, NotShadowCaster);
        world
            .resource(Materials)
            .update(world.storage(MeshMaterial).material.get(bulb), { emissive: [12, 8, 3] });
        const light = world.create();
        world.add(light, Transform, { translation: [-1, 3.85, z] });
        world.add(light, PointLight, {
            color: 0xffd49a,
            intensity: 400000,
            range: 6,
            shadowMapsEnabled: 1,
        });
        scene.pendants.push(light);
        scene.bulbs.push(bulb);
    }
    scene.downlight = world.create();
    world.add(scene.downlight, Transform, {
        translation: [2, 4.7, 7],
        rotation: lookAtRotation(2, 4.7, 7, 2, 0, 7),
    });
    world.add(scene.downlight, SpotLight, {
        color: 0xffe8cc,
        intensity: 200000,
        range: 6,
        innerAngle: 0.3,
        outerAngle: 0.5,
        shadowMapsEnabled: 1,
    });
    const downlightBulb = block(world, [2, 4.75, 7], [0.3, 0.08, 0.3], AMBER);
    world.add(downlightBulb, NotShadowCaster);
    world.resource(Materials).update(world.storage(MeshMaterial).material.get(downlightBulb), {
        emissive: [12, 8, 3],
    });
    scene.bulbs.push(downlightBulb);
    scene.object = block(world, [-1, 0.55, 4], [0.8, 1.1, 0.8], [0.658, 0.301, 0.117, 1]);
    block(world, [2, 0.3, 7], [1.1, 0.6, 1.1]);
    block(world, [2.5, 0.65, 8.2], [0.7, 1.3, 0.7]);
    const sky = world.create();
    world.add(sky, Sky, {
        brightness: 8 * 2 ** 9.7,
        zenith: 0xd8e4ff,
        horizon: 0xeee9df,
        cloudCoverage: 0.2,
    });
    scene.camera = world.create();
    world.add(scene.camera, Camera, { fov: 70, clearColor: 0x191816 });
    world.add(scene.camera, StandardRenderer);
    world.add(scene.camera, CameraBackground, { background: skyBackgroundHandle(world) });
    world.add(scene.camera, Transform, { translation: [2.5, 1.7, 11] });
    scene.player = world.create();
    world.add(scene.player, Transform, { translation: [2.5, 1, 11] });
    world.add(scene.player, Body, { type: BodyType.Kinematic });
    world.add(scene.player, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, -0.3, 0],
        capsuleCenter2: [0, 0.3, 0],
        radius: 0.3,
    });
    world.add(scene.player, Character);
    world.add(scene.player, Player, { camera: scene.camera, yaw: 0.25, pitch: -0.05 });
    world.add(scene.player, LocalPlayer);
}
export const RoomPlugin: Plugin = {
    name: "Room",
    dependencies: [StandardPhysicsPlugin, PlayerPlugin, FogPlugin, SkyPlugin],
    initialize: room,
};
