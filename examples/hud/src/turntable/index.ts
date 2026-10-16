import {
    Camera,
    component,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    type Resource,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";

export const Turntable = component("Turntable", {});

/** Angle in radians and speed in radians per second. */
export const Turn: Resource<{ angle: number; speed: number }> = {
    create: () => ({ angle: 0, speed: 1.2 }),
};

export const CubeMaterial: Resource<{ handle?: MaterialHandle }> = {
    create: () => ({}),
};

function makeScene(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 520;

    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.5, -1, -0.6) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);

    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.7, pitch: 0.32 });
    world.add(camera, Transform);

    const materials = world.resource(Materials);
    const cube = world.create();
    world.add(cube, MeshInstance);
    const handle = materials.add(
        StandardMaterial({
            baseColor: [0.72, 0.4, 0.05, 1],
            perceptualRoughness: 0.38,
        }),
    );
    world.resource(CubeMaterial).handle = handle;
    world.add(cube, MeshMaterial, handle);
    world.add(cube, Transform, { translation: [0, 0.15, 0] });
    world.add(cube, Turntable);

    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(
        ground,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.085, 0.08, 0.072, 1],
                perceptualRoughness: 0.82,
            }),
        ),
    );
    world.add(ground, Transform, { translation: [0, -0.62, 0], scale: [8, 0.2, 8] });
}

const TurnCube = {
    name: "Turntable/Turn",
    update(world: World): void {
        const turn = world.resource(Turn);
        turn.angle += world.time.deltaTime * turn.speed;
        const half = turn.angle * 0.5;
        const rotation = world.storage(Transform).rotation;
        for (const eid of world.query([Turntable]))
            rotation.set(eid, 0, Math.sin(half), 0, Math.cos(half));
    },
};

export const TurntablePlugin: Plugin = {
    name: "Turntable",
    components: [Turntable],
    dependencies: [OrbitPlugin],
    initialize: makeScene,
    systems: [TurnCube],
};
