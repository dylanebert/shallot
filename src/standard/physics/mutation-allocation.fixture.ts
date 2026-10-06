import type { PhysicsWorld } from "./api/world";
import { BodyType, defaultBodyDef, defaultShapeDef } from "./common/types";
import { createSphereShape } from "./shapes/shape";
import { createBody, destroyBody } from "./world/body";

export function mutationAllocationSubject(physics: PhysicsWorld): () => void {
    const a = physics.createBody({ type: BodyType.Dynamic });
    const b = physics.createBody({ type: BodyType.Dynamic, position: { x: 4, y: 0, z: 0 } });
    const shape = a.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
    const joint = physics.createDistanceJoint(a, b);
    const spherical = physics.createSphericalJoint(a, b);
    const wheel = physics.createWheelJoint(a, b);
    const motor = physics.createMotorJoint(a, b);
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
        a.setType(on ? BodyType.Static : BodyType.Dynamic);
        shape.setFilter(filters[+on]);
        shape.enableSensorEvents(on);
        shape.enableContactEvents(on);
        shape.enableHitEvents(on);
        joint.setCollideConnected(on);
        joint.enableMotor(on);
        joint.setLength(2);
        joint.setLengthRange(1, 4);
        spherical.enableSpring(on);
        spherical.enableMotor(on);
        spherical.setTargetRotation(rotation);
        spherical.setMotorVelocity(vector);
        wheel.enableSteering(on);
        wheel.enableSuspension(on);
        wheel.enableSuspensionLimit(on);
        wheel.enableSteeringLimit(on);
        wheel.enableSpinMotor(on);
        wheel.setSuspensionLimits(-1, 1);
        wheel.setSpinMotorSpeed(2);
        wheel.setTargetSteeringAngle(0.1);
        motor.setLinearVelocity(vector);
        motor.setAngularVelocity(vector);
        motor.setMaxSpringForce(-1);
        motor.setMaxSpringTorque(1);
        const id = createBody(physics.state, bodyDef);
        createSphereShape(physics.state, id, shapeDef, sphere);
        destroyBody(physics.state, id);
    };
}
