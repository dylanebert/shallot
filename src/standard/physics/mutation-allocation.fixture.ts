import { BodyType, defaultBodyDef, defaultShapeDef } from "./common/types";
import * as C from "./kernel/columns";
import { writeJointFloat, writeJointQuat, writeJointVec3 } from "./kernel/jointcolumns";
import { kernel } from "./kernel/kernel";
import { ShapeFlags, setShapeFlag } from "./kernel/shaperecords";
import { createSphereShape, destroyShape, setShapeFilter } from "./shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "./solver/distanceJoint";
import { defaultJointDef, setJointCollideConnected, wakeJointBodies } from "./solver/joint";
import { createMotorJoint, defaultMotorJointDef } from "./solver/motorJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "./solver/sphericalJoint";
import { createWheelJoint, defaultWheelJointDef } from "./solver/wheelJoint";
import { bodySetType, createBody, destroyBody } from "./world/body";
import type { WorldState } from "./world/world";

export function mutationAllocationSubject(world: WorldState): () => void {
    const k = kernel(world.ecsState);
    const a = createBody(world, { ...defaultBodyDef(), type: BodyType.Dynamic });
    const b = createBody(world, {
        ...defaultBodyDef(),
        type: BodyType.Dynamic,
        position: { x: 4, y: 0, z: 0 },
    });
    const shape = createSphereShape(world, a, defaultShapeDef(), {
        center: { x: 0, y: 0, z: 0 },
        radius: 1,
    })!;
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    const distance = createDistanceJoint(world, defaultDistanceJointDef(base)).joint;
    const spherical = createSphericalJoint(world, defaultSphericalJointDef(base)).joint;
    const wheel = createWheelJoint(world, defaultWheelJointDef(base)).joint;
    const motor = createMotorJoint(world, defaultMotorJointDef(base)).joint;
    const filters = [
        { categoryBits: 1n, maskBits: 1n, groupIndex: 0 },
        { categoryBits: 2n, maskBits: 2n, groupIndex: 0 },
    ];
    const vector = { x: 0.1, y: 0.2, z: 0.3 };
    const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    const bodyDef = defaultBodyDef();
    bodyDef.type = BodyType.Dynamic;
    const shapeDef = defaultShapeDef();
    const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 1 };
    let frame = 0;
    return () => {
        const on = !!(frame++ & 1);
        bodySetType(world, a, on ? BodyType.Static : BodyType.Dynamic);
        setShapeFilter(world, shape, filters[+on]!);
        setShapeFlag(world, shape, ShapeFlags.enableSensorEvents, on);
        setShapeFlag(world, shape, ShapeFlags.enableContactEvents, on);
        setShapeFlag(world, shape, ShapeFlags.enableHitEvents, on);
        setJointCollideConnected(world, distance, on);
        k.jointEnable(world.worldId, distance, C.DJ_ENABLE_MOTOR, on);
        k.distanceJointSetLength(world.worldId, distance, 2);
        k.jointSetLimits(world.worldId, distance, 1, 4);
        k.jointEnable(world.worldId, spherical, C.SJ_ENABLE_SPRING, on);
        k.jointEnable(world.worldId, spherical, C.SJ_ENABLE_MOTOR, on);
        writeJointQuat(world, spherical, C.SJ_TARGET_ROTATION, rotation);
        writeJointVec3(world, spherical, C.SJ_MOTOR_VELOCITY, vector);
        wakeJointBodies(world, spherical);
        k.jointEnable(world.worldId, wheel, C.WHJ_ENABLE_STEERING, on);
        k.jointEnable(world.worldId, wheel, C.WHJ_ENABLE_SUSPENSION_SPRING, on);
        k.jointEnable(world.worldId, wheel, C.WHJ_ENABLE_SUSPENSION_LIMIT, on);
        k.jointEnable(world.worldId, wheel, C.WHJ_ENABLE_STEERING_LIMIT, on);
        k.jointEnable(world.worldId, wheel, C.WHJ_ENABLE_SPIN_MOTOR, on);
        k.jointSetLimits(world.worldId, wheel, -1, 1);
        writeJointFloat(world, wheel, C.WHJ_SPIN_SPEED, 2);
        wakeJointBodies(world, wheel);
        writeJointFloat(world, wheel, C.WHJ_TARGET_STEERING_ANGLE, 0.1);
        wakeJointBodies(world, wheel);
        writeJointVec3(world, motor, C.MJ_LINEAR_VELOCITY, vector);
        wakeJointBodies(world, motor);
        writeJointVec3(world, motor, C.MJ_ANGULAR_VELOCITY, vector);
        wakeJointBodies(world, motor);
        writeJointFloat(world, motor, C.MJ_MAX_SPRING_FORCE, -1);
        writeJointFloat(world, motor, C.MJ_MAX_SPRING_TORQUE, 1);
        const id = createBody(world, bodyDef);
        const temporary = createSphereShape(world, id, shapeDef, sphere)!;
        destroyShape(world, temporary, true);
        destroyBody(world, id);
    };
}
