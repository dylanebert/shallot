import {
    Body,
    BodyType,
    Camera,
    CameraMode,
    DirectionalLight,
    GlobalAmbientLight,
    GlobalTransform,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    NotShadowCaster,
    type Plugin,
    Shape,
    ShapeKind,
    StandardMaterial,
    StandardRenderer,
    sphere,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import {
    attachTexture,
    RenderLayers,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import { StandardPhysicsPlugin, StepPhysicsSystem } from "@dylanebert/shallot/standard/physics";
import { GrabPlugin } from "./grab";
import { SnowMaterial, SnowPlugin, StampMaterial, snowPlane } from "./snow";
import { DEPTH, FIELD, RESOLUTION, Trail } from "./trail";

export const RADIUS = DEPTH * 2;
export const SnowTrails = {
    create: () => ({ camera: 0, capture: 0, boulder: 0, stamp: 0, snow: 0 }),
};
export const DemoPlugin: Plugin = {
    name: "SnowTrails",
    dependencies: [SnowPlugin, OrbitPlugin, StandardPhysicsPlugin, GrabPlugin],
    initialize(world) {
        const state = world.resource(SnowTrails);
        const rockMesh = world.resource(Meshes).add({ name: "boulder", ...sphere(RADIUS, 48, 32) });
        const rock = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.8 }));
        const boulder = world.create();
        state.boulder = boulder;
        world.add(boulder, Transform, { translation: [-6, RADIUS, 0] });
        world.add(boulder, Body, {
            type: BodyType.Dynamic,
            linearVelocity: [3, 0, 0],
            angularVelocity: [0, 0, -3 / RADIUS],
        });
        world.add(boulder, Shape, { kind: ShapeKind.Sphere, radius: RADIUS, friction: 0.8 });
        world.add(boulder, MeshInstance, { mesh: rockMesh });
        world.add(boulder, MeshMaterial, rock);
        const stamp = world.create();
        state.stamp = stamp;
        world.add(stamp, Transform, { translation: [-6, RADIUS, 0] });
        world.add(stamp, MeshInstance, { mesh: rockMesh });
        world.add(stamp, MeshMaterial, world.resource(StampMaterial).add({ unused: 0 }));
        world.add(stamp, RenderLayers, { mask: 2 });
        const snow = world.create();
        state.snow = snow;
        world.add(snow, Transform);
        world.add(snow, NotShadowCaster);
        world.add(snow, MeshInstance, {
            mesh: world.resource(Meshes).add({ name: "snowfield", ...snowPlane() }),
        });
        world.add(snow, MeshMaterial, world.resource(SnowMaterial).add({ unused: 0 }));
        for (const [translation, scale] of [
            [
                [0, -0.5, 0],
                [FIELD, 1, FIELD],
            ],
            [
                [-12.5, 1, 0],
                [1, 3, FIELD + 2],
            ],
            [
                [12.5, 1, 0],
                [1, 3, FIELD + 2],
            ],
            [
                [0, 1, -12.5],
                [FIELD, 3, 1],
            ],
            [
                [0, 1, 12.5],
                [FIELD, 3, 1],
            ],
        ] as const) {
            const body = world.create();
            world.add(body, Transform, { translation: [...translation] });
            world.add(body, Body, { type: BodyType.Static });
            world.add(body, Shape, { scale: [...scale] });
        }
        const capture = world.create();
        state.capture = capture;
        world.add(capture, Transform, { rotation: lookAtRotation(0, 0, 0, 0, 1, 0, 0, 0, 1) });
        world.add(capture, Camera, {
            mode: CameraMode.Orthographic,
            size: FIELD / 2,
            near: 0,
            far: DEPTH,
            clearColor: 0,
            antialias: 0,
            order: -1,
        });
        world.add(capture, StandardRenderer);
        world.add(capture, RenderLayers, { mask: 2 });
        world.add(capture, Tonemapping, { method: TonemappingMethod.None });
        world.resource(Trail).capture = attachTexture(world, capture, {
            width: RESOLUTION,
            height: RESOLUTION,
        });
        const camera = world.create();
        state.camera = camera;
        world.add(camera, Transform);
        world.add(camera, Camera, { clearColor: 0x191816, far: 100 });
        world.add(camera, StandardRenderer);
        world.add(camera, Orbit, {
            distance: 27,
            yaw: 0.35,
            pitch: 0.85,
            minDistance: 10,
            maxDistance: 45,
        });
        Object.assign(world.resource(GlobalAmbientLight), { color: 0xd8e4ff, brightness: 380 });
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 6000,
            shadowMapsEnabled: 1,
        });
    },
    systems: [
        {
            name: "boulder-stamp",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const { boulder, stamp } = world.resource(SnowTrails);
                const pose = world.storage(GlobalTransform);
                world
                    .storage(Transform)
                    .translation.set(
                        stamp,
                        pose.translation.x.get(boulder),
                        pose.translation.y.get(boulder),
                        pose.translation.z.get(boulder),
                    );
                world
                    .storage(Transform)
                    .rotation.set(
                        stamp,
                        pose.rotation.x.get(boulder),
                        pose.rotation.y.get(boulder),
                        pose.rotation.z.get(boulder),
                        pose.rotation.w.get(boulder),
                    );
            },
        },
    ],
};
