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
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";

export const WelcomeCube = component("WelcomeCube", {});

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
    world.add(cube, WelcomeCube);

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
}

const SpinCube = {
    name: "Welcome/SpinCube",
    update(world: World): void {
        const halfAngle = world.time.deltaTime * 0.65 * 0.5;
        if (halfAngle === 0) return;
        const sine = Math.sin(halfAngle);
        const cosine = Math.cos(halfAngle);
        const rotation = world.storage(Transform).rotation;
        for (const eid of world.query([WelcomeCube])) {
            const y = rotation.y.get(eid);
            const w = rotation.w.get(eid);
            rotation.set(eid, 0, y * cosine + w * sine, 0, w * cosine - y * sine);
        }
    },
};

export const Welcome: Plugin = {
    name: "Welcome",
    components: [WelcomeCube],
    dependencies: [OrbitPlugin],
    initialize: makeScene,
    systems: [SpinCube],
};
