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

/** Radians per second. */
const SPEED = 1.2;

export const Turntable = component("Turntable", {});

/** How far the turntable has turned: `angle` in radians, `turns` completed. */
export const Turn: Resource<{ angle: number; turns: number }> = {
    create: () => ({ angle: 0, turns: 0 }),
};

function makeScene(world: World): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 520;

    const sun = world.create();
    const direction = lookAtRotation(0, 0, 0, -0.5, -1, -0.6);
    world.add(sun, Transform, {
        rotation: [direction.x, direction.y, direction.z, direction.w],
    });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);

    const camera = world.create();
    world.add(camera, Camera, { clearColor: 0x1c1917 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, { distance: 5, yaw: 0.7, pitch: 0.32 });
    world.add(camera, Transform);

    const materials = world.resource(Materials);
    const cube = world.create();
    world.add(cube, MeshInstance);
    world.add(
        cube,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.621407, 0.26791815, 0.11756791, 1],
                perceptualRoughness: 0.38,
            }),
        ),
    );
    world.add(cube, Transform, { translation: [0, 0.15, 0, 0] });
    world.add(cube, Turntable);

    const ground = world.create();
    world.add(ground, MeshInstance);
    world.add(
        ground,
        MeshMaterial,
        materials.add(
            StandardMaterial({
                baseColor: [0.38498746, 0.3374035, 0.26791645, 1],
                perceptualRoughness: 0.82,
            }),
        ),
    );
    world.add(ground, Transform, { translation: [0, -0.62, 0, 0], scale: [8, 0.2, 8, 0] });
}

const TurnCube = {
    name: "Turntable/Turn",
    update(world: World): void {
        const turn = world.resource(Turn);
        turn.angle += world.time.deltaTime * SPEED;
        turn.turns = Math.floor(turn.angle / (2 * Math.PI));
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
