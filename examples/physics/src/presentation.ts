import {
    Body,
    BodyType,
    cube,
    DistanceJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    StandardMaterial,
    shapeEdges,
    shapeMesh,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";
import { isAwake } from "@dylanebert/shallot/standard/physics";
import { EventSample } from "./events";
import { MotorizedJoint } from "./roles";
import { controlValues, enabledControls, type Scene, scenes } from "./scenes";
import { SensorSample } from "./sensor";

const materials = {
    ground: StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    subject: StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    motorized: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
    sensor: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
};

const sleeping = {
    ground: materials.ground,
    subject: StandardMaterial({ baseColor: [0.52, 0.29, 0.065, 1], perceptualRoughness: 0.9 }),
    motorized: StandardMaterial({ baseColor: [0.055, 0.29, 0.35, 1], perceptualRoughness: 0.9 }),
    sensor: StandardMaterial({ baseColor: [0.055, 0.29, 0.35, 1], perceptualRoughness: 0.9 }),
};
const reaction = StandardMaterial({ baseColor: [1, 0.22, 0.035, 1], perceptualRoughness: 0.3 });
const restingReaction = StandardMaterial({
    baseColor: [0.65, 0.16, 0.045, 1],
    perceptualRoughness: 0.9,
});

export function shapeMaterial(world: World, role: keyof typeof materials, eid: number) {
    if (role === "ground") return materials.ground;
    const body = world.storage(Shape).body.get(eid) || eid;
    // Hue marks a reaction; resting bodies keep their role's tint but lose saturation and gloss.
    const asleep = isAwake(world, body) === false;
    if (world.resource(SensorSample).sensors.get(world.ref(eid))?.row === 20) return reaction;
    if (
        world.resource(EventSample).reactions.has(world.ref(body)) ||
        world.resource(SensorSample).reactions.has(world.ref(body))
    )
        return asleep ? restingReaction : reaction;
    return asleep ? sleeping[role] : materials[role];
}

export function drivenBodies(world: World, bodies = new Set<number>()): Set<number> {
    bodies.clear();
    for (const joint of [DistanceJoint, PrismaticJoint, RevoluteJoint, WheelJoint]) {
        const storage = world.storage(joint);
        for (const eid of world.query([joint, MotorizedJoint])) {
            bodies.add(storage.a.get(eid));
            bodies.add(storage.b.get(eid));
        }
    }
    return bodies;
}

export function shapeRole(
    world: World,
    eid: number,
    driven: ReadonlySet<number>,
): keyof typeof materials {
    const shape = world.storage(Shape);
    const body = shape.body.get(eid) || eid;
    const type = world.storage(Body).type.get(body);
    if (world.resource(SensorSample).sensors.get(world.ref(eid))?.active) return "ground";
    if (shape.isSensor.get(eid)) return "sensor";
    if (type === BodyType.Static) return "ground";
    if (
        type === BodyType.Kinematic ||
        body === world.resource(EventSample).player ||
        driven.has(body)
    )
        return "motorized";
    return "subject";
}

type Geometry = {
    mesh: ReturnType<typeof shapeMesh> & { name: string };
    edges: ReturnType<typeof shapeEdges>;
};
export type ShapeGeometryCache = Map<string, Geometry>;

export function shapePresentation(
    world: World,
    eid: number,
    cache?: ShapeGeometryCache,
    driven: ReadonlySet<number> = drivenBodies(world),
) {
    const shape = world.get(eid, Shape)!;
    const body = shape.body || eid;
    const role = shapeRole(world, eid, driven);
    const key = JSON.stringify([
        shape.kind,
        shape.geometry,
        shape.scale,
        shape.radius,
        shape.sphereCenter,
        shape.capsuleCenter1,
        shape.capsuleCenter2,
        shape.hullPosition,
        shape.hullRotation,
    ]);
    let geometry = cache?.get(key);
    if (!geometry) {
        geometry = {
            mesh: { name: `Shape ${eid}`, ...shapeMesh(world, shape) },
            edges: shapeEdges(world, shape),
        };
        cache?.set(key, geometry);
    }
    return { eid, body, role, ...geometry };
}

function destructionRings() {
    const lines: { a: [number, number, number]; b: [number, number, number]; color: number }[] = [];
    for (const [radius, color] of [
        [2.5, 0x00ffff],
        [3.75, 0xfff8dc],
    ] as const)
        for (let axis = 0; axis < 3; axis++)
            for (let i = 0; i < 24; i++) {
                const point = (index: number): [number, number, number] => {
                    const p: [number, number, number] = [0, 5, 0];
                    const angle = (index * 2 * Math.PI) / 24;
                    p[(axis + 1) % 3]! += radius * Math.cos(angle);
                    p[(axis + 2) % 3]! += radius * Math.sin(angle);
                    return p;
                };
                lines.push({ a: point(i), b: point(i + 1), color });
            }
    return lines;
}

export function scenePresentation(
    world: World,
    scene: Scene,
    cache: ShapeGeometryCache = new Map(),
) {
    const driven = drivenBodies(world);
    const shapes = Array.from(scene === "largeWorld" ? [] : world.query([Shape]), (eid) =>
        shapePresentation(world, eid, cache, driven),
    );
    const { gridExtent: extent, gridSpacing: spacing } = scenes[scene];
    const lines: { a: [number, number, number]; b: [number, number, number]; color: number }[] = [];
    for (let offset = -extent; offset <= extent; offset += spacing) {
        const color = offset === 0 ? 0x797369 : 0x49453f;
        lines.push(
            { a: [-extent, 0.005, offset], b: [extent, 0.005, offset], color },
            { a: [offset, 0.005, -extent], b: [offset, 0.005, extent], color },
        );
    }
    return {
        shapes,
        grid: { lines, width: 1, opacity: 0.3 },
        explosion: scene === "destruction" ? destructionRings() : [],
        ground:
            scene === "sensor"
                ? {
                      mesh: { name: "Ground", ...cube([20, 0.2, 20]) },
                      translation: [0, -0.1, 0] as [number, number, number],
                  }
                : undefined,
        materials,
        controls: enabledControls(scene, controlValues(scene)),
    };
}
