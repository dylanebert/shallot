import {
    Body,
    BodyType,
    Shape,
    StandardMaterial,
    shapeMesh,
    type World,
} from "@dylanebert/shallot";
import { controlValues, type Scene, visibleControls } from "./scenes";

export function scenePresentation(world: World, scene: Scene) {
    const materials = {
        ground: StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
        subject: StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
        motorized: StandardMaterial({
            baseColor: [0.04, 0.43, 0.53, 1],
            perceptualRoughness: 0.38,
        }),
    };
    const shapes = [];
    for (const eid of world.query([Shape])) {
        const shape = world.get(eid, Shape)!;
        const body = shape.body || eid;
        // Static bodies are carbon, grabbable dynamics amber, and driven joints cyan.
        const role: keyof typeof materials =
            world.storage(Body).type.get(body) === BodyType.Static
                ? "ground"
                : scene === "revolute" || scene === "prismatic" || scene === "wheel"
                  ? "motorized"
                  : "subject";
        shapes.push({
            eid,
            body,
            role,
            mesh: { name: `Shape ${eid}`, ...shapeMesh(world, shape) },
        });
    }
    return { shapes, materials, controls: visibleControls(scene, controlValues(scene)) };
}
