import {
    Camera,
    component,
    DirectionalLight,
    drawLine,
    f32,
    Gizmo,
    GizmoAsset,
    GlobalAmbientLight,
    internText,
    LinesPlugin,
    linearToSrgb,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    Orbit,
    OrbitPlugin,
    type Plugin,
    type Resource,
    registerFont,
    registerGizmo,
    registerImage,
    Sprite,
    SpriteFill,
    SpritePlugin,
    StandardMaterial,
    StandardRenderer,
    Text,
    TextPlugin,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Anchor, Billboard } from "@dylanebert/shallot/rendering";

type Vec3 = readonly [number, number, number];
type Color = readonly [number, number, number, number];
const GROUND = [0.085, 0.08, 0.072, 1] as const;
const CYAN = [0.04, 0.43, 0.53, 1] as const;
const AMBER = [0.72, 0.4, 0.05, 1] as const;
const DARK = [0.042, 0.039, 0.035, 1] as const;
export const ROAD_START = -4.5;
export const ROAD_LENGTH = 7.5;
export const DRIVE_SECONDS = 6;
export const EXIT_SECONDS = 0.35;
export const CYCLE_SECONDS = 8;
export const Car = component("Delivery/Car", { progress: f32 });
export const Delivery: Resource<{
    camera: number;
    car: number;
    label: number;
    bar: number;
    backing: number;
    icon: number;
    font: number;
    image: number;
    white: number;
    parts: { eid: number; offset: Vec3 }[];
    trail: number;
    trailHandle: number;
    trailStart: number[];
    trailEnd: number[];
}> = {
    create: () => ({
        camera: 0,
        car: 0,
        label: 0,
        bar: 0,
        backing: 0,
        icon: 0,
        font: 0,
        image: 0,
        white: 0,
        parts: [],
        trail: 0,
        trailHandle: 0,
        trailStart: [ROAD_START, 0.16, 1],
        trailEnd: [ROAD_START, 0.16, 1],
    }),
};
export function assets(world: World): void {
    const scene = world.resource(Delivery);
    scene.font = registerFont(world, new URL("../public/Inter-Regular.ttf", import.meta.url).href);
    scene.image = registerImage(world, new URL("../public/attention.png", import.meta.url).href);
    scene.white = registerImage(world, new URL("../public/white.png", import.meta.url).href);
}
function block(world: World, position: Vec3, scale: Vec3, color: Color): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, scale });
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: color, perceptualRoughness: 0.65 })),
    );
    return eid;
}
function scene(world: World): void {
    const scene = world.resource(Delivery);
    Object.assign(world.resource(GlobalAmbientLight), { color: 0xd8e4ff, brightness: 380 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000, shadowMapsEnabled: 1 });
    scene.camera = world.create();
    world.add(scene.camera, Camera, { clearColor: 0x191816 });
    world.add(scene.camera, StandardRenderer);
    world.add(scene.camera, Transform);
    world.add(scene.camera, Orbit, { pan: [0, 0.4, 0], distance: 13, yaw: 0.35, pitch: 0.85 });
    block(world, [0, -0.2, 0], [12, 0.4, 8], GROUND);
    block(world, [0, 0.015, 1], [10, 0.03, 1.3], [0.15, 0.14, 0.125, 1]);
    block(world, [3, 0.9, -2], [2.2, 1.8, 1.8], AMBER);
    block(world, [3, 1.86, -2], [2.05, 0.12, 1.65], [0.06, 0.057, 0.052, 1]);
    block(world, [3, 0.35, -1.09], [0.35, 0.7, 0.02], DARK);
    for (const x of [2.3, 3.7])
        block(world, [x, 1.1, -1.08], [0.35, 0.3, 0.02], [0.11, 0.105, 0.095, 1]);
    scene.icon = world.create();
    world.add(scene.icon, Transform, { translation: [3, 3.1, -2] });
    world.add(scene.icon, Sprite, {
        image: scene.image,
        size: [0.95, 0.95],
        billboard: Billboard.Screen,
    });
    world.add(scene.icon, Anchor, { offset: Anchor.CENTER });
    world.add(scene.icon, NotShadowCaster);
    scene.car = block(world, [ROAD_START, 0.32, 1], [1.1, 0.32, 0.65], CYAN);
    world.add(scene.car, Car);
    scene.parts.push({ eid: scene.car, offset: [0, 0.32, 1] });
    const cab = block(world, [ROAD_START - 0.1, 0.62, 1], [0.55, 0.32, 0.58], CYAN);
    scene.parts.push({ eid: cab, offset: [-0.1, 0.62, 1] });
    const wheels: Vec3[] = [
        [-0.35, 0.15, 0.65],
        [0.35, 0.15, 0.65],
        [-0.35, 0.15, 1.35],
        [0.35, 0.15, 1.35],
    ];
    for (const offset of wheels) {
        const eid = block(
            world,
            [ROAD_START + offset[0], offset[1], offset[2]],
            [0.24, 0.28, 0.14],
            DARK,
        );
        scene.parts.push({ eid, offset });
    }
    scene.label = world.create();
    world.add(scene.label, Transform);
    world.add(scene.label, Text, {
        content: internText(world, "Courier"),
        font: scene.font,
        fontSize: 0.5,
        billboard: Billboard.Screen,
        color: 0xeee9df,
    });
    world.add(scene.label, Anchor);
    scene.backing = world.create();
    world.add(scene.backing, Transform);
    world.add(scene.backing, Sprite, {
        image: scene.white,
        size: [1.85, 0.26],
        color: GROUND.slice(0, 3).reduce(
            (rgb, channel) => (rgb << 8) | Math.round(linearToSrgb(channel) * 255),
            0,
        ),
        billboard: Billboard.Screen,
    });
    world.add(scene.backing, NotShadowCaster);
    scene.bar = world.create();
    world.add(scene.bar, Transform);
    world.add(scene.bar, Sprite, {
        image: scene.white,
        size: [1.7, 0.14],
        color: CYAN.slice(0, 3).reduce(
            (rgb, channel) => (rgb << 8) | Math.round(linearToSrgb(channel) * 255),
            0,
        ),
        billboard: Billboard.Screen,
        fillMethod: SpriteFill.Horizontal,
    });
    world.add(scene.bar, NotShadowCaster);
    const trail = new GizmoAsset();
    drawLine(trail, [0, 0, 0], [1, 0, 0], 0xeee9df);
    scene.trailHandle = registerGizmo(world, trail);
    scene.trail = world.create();
    world.add(scene.trail, Transform);
    DriveCar.update(world);
}
export const DriveCar = {
    name: "Delivery/drive",
    group: "simulation" as const,
    update(world: World): void {
        const scene = world.resource(Delivery);
        const phase = world.time.elapsed % CYCLE_SECONDS;
        const progress = Math.min(phase / DRIVE_SECONDS, 1);
        const x = ROAD_START + ROAD_LENGTH * progress;
        world.storage(Car).progress.set(scene.car, progress);
        const transforms = world.storage(Transform);
        const positions = transforms.translation;
        for (const part of scene.parts)
            positions.set(part.eid, x + part.offset[0], part.offset[1], part.offset[2]);
        positions.set(scene.label, x, 1.95, 1);
        positions.set(scene.backing, x, 1.3, 1);
        positions.set(scene.bar, x, 1.3, 1.03);
        world.storage(Sprite).fillAmount.set(scene.bar, progress);
        scene.trailEnd[0] = x - transforms.scale.x.get(scene.car) / 2;
        scene.trailEnd[1] = positions.y.get(scene.car) - transforms.scale.y.get(scene.car) / 2;
        scene.trailStart[1] = scene.trailEnd[1];
        const length = scene.trailEnd[0] - scene.trailStart[0];
        positions.set(scene.trail, scene.trailStart[0], scene.trailStart[1], scene.trailStart[2]);
        transforms.scale.set(scene.trail, Math.max(0, length), 1, 1);
        if (length > 0) {
            if (!world.has(scene.trail, Gizmo))
                world.add(scene.trail, Gizmo, { handle: scene.trailHandle, width: 2 });
        } else if (world.has(scene.trail, Gizmo)) world.remove(scene.trail, Gizmo);
        const t = Math.max(0, Math.min(1, (phase - DRIVE_SECONDS) / EXIT_SECONDS));
        const c1 = 1.70158;
        const easeInBack = t * t * ((c1 + 1) * t - c1);
        const scale = t < 1 ? 1 - easeInBack : 0;
        transforms.scale.set(scene.icon, scale, scale, scale);
    },
};
export const DeliveryPlugin: Plugin = {
    name: "Delivery",
    components: [Car],
    dependencies: [OrbitPlugin, TextPlugin, SpritePlugin, LinesPlugin],
    initialize: scene,
    systems: [DriveCar],
};
