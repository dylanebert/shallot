import {
    Body,
    BodyType,
    cube,
    Shape,
    StandardMaterial,
    shapeEdges,
    shapeMesh,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { isAwake } from "@dylanebert/shallot/standard/physics";
import { EventSample } from "./events";
import { controlValues, type Scene, visibleControls } from "./scenes";

const materials = {
    ground: StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    subject: StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    motorized: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
    sensor: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
};

const sleeping = {
    ground: materials.ground,
    subject: StandardMaterial({ baseColor: [0.48, 0.42, 0.3, 1], perceptualRoughness: 0.9 }),
    motorized: StandardMaterial({ baseColor: [0.29, 0.43, 0.46, 1], perceptualRoughness: 0.9 }),
    sensor: StandardMaterial({ baseColor: [0.29, 0.43, 0.46, 1], perceptualRoughness: 0.9 }),
};
const reaction = StandardMaterial({ baseColor: [1, 0.22, 0.035, 1], perceptualRoughness: 0.3 });
const restingReaction = StandardMaterial({
    baseColor: [0.53, 0.34, 0.26, 1],
    perceptualRoughness: 0.9,
});

export function shapeMaterial(world: World, role: keyof typeof materials, eid: number) {
    if (role === "ground") return materials.ground;
    const body = world.storage(Shape).body.get(eid) || eid;
    // Hue marks a reaction; resting bodies keep their role's tint but lose saturation and gloss.
    const asleep = isAwake(world, body) === false;
    if (world.resource(EventSample).reactions.has(world.ref(body)))
        return asleep ? restingReaction : reaction;
    return asleep ? sleeping[role] : materials[role];
}

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
    return {
        eid,
        body,
        role,
        mesh: { name: `Shape ${eid}`, ...shapeMesh(world, shape) },
        edges: shapeEdges(world, shape),
    };
}

export function scenePresentation(world: World, scene: Scene) {
    const shapes = Array.from(world.query([Shape]), (eid) => shapePresentation(world, scene, eid));
    let extent = scene === "sensor" ? 10 : 0;
    for (const shape of shapes) {
        if (shape.role !== "ground") continue;
        const pose = world.get(shape.body, Transform);
        const vertices = shape.mesh.vertices;
        let top = -Infinity;
        for (let i = 1; i < vertices.length; i += 8) {
            top = Math.max(top, vertices[i]! * (pose?.scale[1] ?? 1) + (pose?.translation[1] ?? 0));
        }
        if (Math.abs(top) > 0.0001) continue;
        for (let i = 0; i < vertices.length; i += 8) {
            const y = vertices[i + 1]! * (pose?.scale[1] ?? 1) + (pose?.translation[1] ?? 0);
            if (Math.abs(y) > 0.0001) continue;
            for (const axis of [0, 2]) {
                extent = Math.max(
                    extent,
                    Math.abs(
                        vertices[i + axis]! * (pose?.scale[axis] ?? 1) +
                            (pose?.translation[axis] ?? 0),
                    ),
                );
            }
        }
    }
    const lines: { a: [number, number, number]; b: [number, number, number]; color: number }[] = [];
    for (let offset = -Math.floor(extent); offset <= extent; offset++) {
        const color = offset === 0 ? 0x797369 : 0x49453f;
        lines.push(
            { a: [-extent, 0.005, offset], b: [extent, 0.005, offset], color },
            { a: [offset, 0.005, -extent], b: [offset, 0.005, extent], color },
        );
    }
    return {
        shapes,
        grid: { lines, width: 1, opacity: 0.3 },
        ground:
            scene === "sensor"
                ? {
                      mesh: { name: "Ground", ...cube([20, 0.2, 20]) },
                      translation: [0, -0.1, 0] as [number, number, number],
                  }
                : undefined,
        materials,
        controls: visibleControls(scene, controlValues(scene)),
    };
}
