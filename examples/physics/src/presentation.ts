import {
    Body,
    BodyType,
    Shape,
    StandardMaterial,
    shapeMesh,
    type World,
} from "@dylanebert/shallot";
import { EventSample } from "./events";
import { controlValues, type Scene, visibleControls } from "./scenes";

const materials = {
    ground: StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    subject: StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    motorized: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
    sensor: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 0.2], perceptualRoughness: 0.38 }),
};

export function shapePresentation(world: World, scene: Scene, eid: number) {
    const shape = world.get(eid, Shape)!;
    const body = shape.body || eid;
    // Carbon is static, amber grabbable; cyan marks sensors, the player and motorized bodies.
    const role: keyof typeof materials = shape.isSensor
        ? "sensor"
        : world.storage(Body).type.get(body) === BodyType.Static
          ? "ground"
          : scene === "revolute" ||
              scene === "prismatic" ||
              scene === "wheel" ||
              body === world.resource(EventSample).player
            ? "motorized"
            : "subject";
    return { eid, body, role, mesh: { name: `Shape ${eid}`, ...shapeMesh(world, shape) } };
}

export function scenePresentation(world: World, scene: Scene) {
    return {
        shapes: Array.from(world.query([Shape]), (eid) => shapePresentation(world, scene, eid)),
        materials,
        controls: visibleControls(scene, controlValues(scene)),
    };
}
