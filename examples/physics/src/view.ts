import {
    Camera,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    MeshInstance,
    MeshMaterial,
    Shape,
    StandardMaterial,
    StandardRenderer,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { type Scene, views } from "./scenes";

export function createView(world: World, scene: Scene): void {
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 380;
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 6000, shadowMapsEnabled: 1 });
    const camera = world.create();
    const view = views[scene];
    world.add(camera, Camera, { clearColor: 0x191816 });
    world.add(camera, StandardRenderer);
    world.add(camera, Orbit, {
        orbitButton: 2,
        yaw: (view.yaw * Math.PI) / 180,
        pitch: (view.pitch * Math.PI) / 180,
        distance: view.distance,
        pan: view.target,
    });
    world.add(camera, Transform);
    const materials = world.resource(Materials);
    const ground = materials.add(
        StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    );
    const subject = materials.add(
        StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    );
    const scale = world.storage(Transform).scale;
    for (const eid of world.query([Shape])) {
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, scale.y.get(eid) === 2 ? ground : subject);
    }
}
