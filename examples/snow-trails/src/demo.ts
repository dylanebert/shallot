import {
    Body,
    BodyType,
    Camera,
    CameraMode,
    DirectionalLight,
    GlobalAmbientLight,
    GlobalTransform,
    HeightFields,
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
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";
import { GrabPlugin, UNGRABBABLE } from "./grab";
import { FIELD, groundAt, HIGH, HILL_SIZE, heights, LOW } from "./hills";
import { SnowMaterial, SnowPlugin, StampMaterial, snowPlane } from "./snow";
import { DEPTH, RESOLUTION, Trail } from "./trail";

export const RADIUS = DEPTH * 2;
export const SnowTrails = {
    create: () => ({
        camera: 0,
        capture: 0,
        boulder: 0,
        stamp: 0,
        stampMaterial: null as MaterialHandle | null,
        /** The boulder's travel since the capture last drew the stamp, in meters. */
        sweep: [0, 0, 0] as [number, number, number],
        snow: 0,
    }),
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
        world.add(boulder, Transform, { translation: [-7, groundAt(-7, 2) + RADIUS + 2, 2] });
        world.add(boulder, Body, {
            type: BodyType.Dynamic,
        });
        world.add(boulder, Shape, { kind: ShapeKind.Sphere, radius: RADIUS, friction: 0.8 });
        world.add(boulder, MeshInstance, { mesh: rockMesh });
        world.add(boulder, MeshMaterial, rock);
        const stamp = world.create();
        state.stamp = stamp;
        world.add(stamp, Transform, { translation: [-7, groundAt(-7, 2) + RADIUS + 2, 2] });
        world.add(stamp, MeshInstance, { mesh: rockMesh });
        state.stampMaterial = world.resource(StampMaterial).add({ sweep: [0, 0, 0] });
        world.add(stamp, MeshMaterial, state.stampMaterial);
        world.add(stamp, RenderLayers, { mask: 2 });
        const snow = world.create();
        state.snow = snow;
        world.add(snow, Transform);
        world.add(snow, NotShadowCaster);
        world.add(snow, MeshInstance, {
            mesh: world.resource(Meshes).add({ name: "snowfield", ...snowPlane() }),
        });
        world.add(snow, MeshMaterial, world.resource(SnowMaterial).add({ sweep: [0, 0, 0] }));
        const ground = world.create();
        world.add(ground, Transform, { translation: [-FIELD / 2, 0, -FIELD / 2] });
        world.add(ground, Body, { type: BodyType.Static });
        world.add(ground, Shape, {
            kind: ShapeKind.HeightField,
            geometry: world.resource(HeightFields).add({
                name: "soft hills",
                countX: HILL_SIZE,
                countZ: HILL_SIZE,
                scale: [FIELD / (HILL_SIZE - 1), 1, FIELD / (HILL_SIZE - 1)],
                heights,
                globalMinimumHeight: LOW,
                globalMaximumHeight: HIGH,
            }),
            friction: 0.8,
        });
        for (const [x, z, width, length] of [
            [-1, 0, 1, FIELD + 2],
            [1, 0, 1, FIELD + 2],
            [0, -1, FIELD + 2, 1],
            [0, 1, FIELD + 2, 1],
        ]) {
            const wall = world.create();
            world.add(wall, Transform, {
                translation: [(x * (FIELD + 1)) / 2, HIGH, (z * (FIELD + 1)) / 2],
            });
            world.add(wall, Body, { type: BodyType.Static });
            world.add(wall, Shape, {
                scale: [width, 40, length],
                filterCategoryLow: UNGRABBABLE,
                filterCategoryHigh: 0,
            });
        }
        const capture = world.create();
        state.capture = capture;
        world.add(capture, Transform, {
            translation: [0, LOW, 0],
            rotation: lookAtRotation(0, 0, 0, 0, 1, 0, 0, 0, 1),
        });
        world.add(capture, Camera, {
            mode: CameraMode.Orthographic,
            size: FIELD / 2,
            near: 0,
            far: HIGH + DEPTH - LOW,
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
                const { boulder, stamp, sweep } = world.resource(SnowTrails);
                const pose = world.storage(GlobalTransform);
                const v = pose.linearVelocity;
                const dt = world.time.deltaTime;
                sweep[0] += v.x.get(boulder) * dt;
                sweep[1] += v.y.get(boulder) * dt;
                sweep[2] += v.z.get(boulder) * dt;
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
        {
            // The capture draws the stamp once a frame; sweeping it back over the ticks' travel since
            // the last frame leaves no gap between frames.
            name: "stamp-sweep",
            update(world) {
                const { stampMaterial, sweep } = world.resource(SnowTrails);
                world.resource(StampMaterial).update(stampMaterial!, { sweep });
                sweep.fill(0);
            },
        },
    ],
};
