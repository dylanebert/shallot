import {
    BaseColor,
    Body,
    BodyType,
    DistanceJoint,
    type EntityRef,
    MeshMaterial,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    StandardMaterial,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";
import { DynamicMoverState } from "./dynamic-mover-scene";
import { EventSample } from "./events";
import { GeometricMoverState } from "./geometric-mover";
import { RigidBodyState } from "./rigid-body";
import { MotorizedJoint } from "./roles";
import { SensorSample } from "./sensor";

export const materials = {
    ground: StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1], perceptualRoughness: 0.82 }),
    subject: StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.38 }),
    motorized: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
    sensor: StandardMaterial({ baseColor: [0.04, 0.43, 0.53, 1], perceptualRoughness: 0.38 }),
};
const reaction = StandardMaterial({ baseColor: [1, 0.22, 0.035, 1], perceptualRoughness: 0.38 });

export const ShapeMaterials = {
    create: () => new Map<keyof typeof materials, MaterialHandle>(),
};

export function setReaction(
    world: World,
    reactions: Set<EntityRef>,
    ref: EntityRef,
    active = true,
): void {
    if (reactions.has(ref) === active) return;
    if (active) reactions.add(ref);
    else reactions.delete(ref);
    const body = world.resolve(ref);
    if (body) refreshShapeColors(world, body);
}

export function refreshShapeColors(world: World, body?: number): void {
    const driven = drivenBodies(world);
    for (const eid of world.query([Shape, BaseColor])) {
        if (body !== undefined && (world.storage(Shape).body.get(eid) || eid) !== body) continue;
        const role = shapeRole(world, eid, driven);
        const color = shapeMaterial(world, role, eid).baseColor;
        const previous = world.get(eid, BaseColor)!.color;
        if (color.some((value, lane) => Math.fround(value) !== previous[lane]))
            world.storage(BaseColor).color.set(eid, color[0], color[1], color[2], color[3]);
        const handle = world.resource(ShapeMaterials).get(role);
        if (handle && world.has(eid, MeshMaterial)) {
            const current = world.get(eid, MeshMaterial)!;
            if (current.material !== handle.material || current.type !== handle.type)
                world.add(eid, MeshMaterial, handle);
        }
    }
}

export function shapeMaterial(world: World, role: keyof typeof materials, eid: number) {
    if (role === "ground") return materials.ground;
    const body = world.storage(Shape).body.get(eid) || eid;
    if (world.resource(SensorSample).sensors.get(world.ref(eid))?.row === 20) return reaction;
    if (
        world.resource(EventSample).reactions.has(world.ref(body)) ||
        world.resource(SensorSample).reactions.has(world.ref(body))
    )
        return reaction;
    return materials[role];
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
    const mover = world.resource(GeometricMoverState);
    if (mover.active && (eid === mover.ignore || mover.policies.get(eid)?.clipVelocity === false))
        return "motorized";
    if (type === BodyType.Static)
        return mover.active && mover.policies.has(eid) ? "sensor" : "ground";
    if (
        type === BodyType.Kinematic ||
        body === world.resource(EventSample).player ||
        body === world.resource(RigidBodyState).character.body ||
        body === world.resource(DynamicMoverState).entity ||
        driven.has(body)
    )
        return "motorized";
    return "subject";
}
