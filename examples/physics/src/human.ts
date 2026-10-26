// SPDX-License-Identifier: MIT
import {
    Body,
    BodyType,
    FilterJoint,
    RevoluteJoint,
    Shape,
    ShapeKind,
    SphericalJoint,
    Transform,
    type World,
} from "@dylanebert/shallot";
import bones from "./human.json";

const f = Math.fround;
const degrees = f(0.01745329251);
function normalized(values: number[]): [number, number, number, number] {
    const [x, y, z, w] = values.map(f) as [number, number, number, number];
    const length = f(Math.sqrt(f(f(f(f(x * x) + f(y * y)) + f(z * z)) + f(w * w))));
    const inv = f(1 / length);
    return [f(inv * x), f(inv * y), f(inv * z), f(inv * w)];
}
/** shared/human.c's capsule bodies and constrained bones. */
export function human(
    world: World,
    position: [number, number, number],
    group: number,
): { bodies: number[]; joints: number[] } {
    const bodies = bones.map((bone) => {
        const eid = world.create();
        world.add(eid, Transform, {
            translation: position.map((v, i) => f(v + f(bone.refP[i]!))) as [
                number,
                number,
                number,
            ],
            rotation: bone.refQ as [number, number, number, number],
        });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: bone.cap1 as [number, number, number],
            capsuleCenter2: bone.cap2 as [number, number, number],
            radius: bone.radius,
            rollingResistance: 0.2,
            filterGroupIndex: bone.negGroup ? -group : 0,
        });
        return eid;
    });
    const joints: number[] = [];
    for (let i = 1; i < bones.length; i++) {
        const bone = bones[i]!;
        const eid = world.create();
        const base = {
            a: bodies[bone.parent]!,
            b: bodies[i]!,
            localAnchorA: bone.lfaP! as [number, number, number],
            localAnchorB: bone.lfbP! as [number, number, number],
            localRotationA: normalized(bone.lfaQ!),
            localRotationB: normalized(bone.lfbQ!),
            enableSpring: 1,
            hertz: 1,
            dampingRatio: 0.7,
            enableMotor: 1,
            maxMotorTorque: f(f(bone.friction) * 5),
        };
        const lower = f(f(bone.twistDeg![0]!) * degrees),
            upper = f(f(bone.twistDeg![1]!) * degrees);
        if (bone.joint === "revolute")
            world.add(eid, RevoluteJoint, {
                ...base,
                enableLimit: 1,
                lowerAngle: lower,
                upperAngle: upper,
            });
        else
            world.add(eid, SphericalJoint, {
                ...base,
                enableConeLimit: 1,
                coneAngle: f(f(bone.swingDeg!) * degrees),
                enableTwistLimit: 1,
                lowerTwistAngle: lower,
                upperTwistAngle: upper,
            });
        joints.push(eid);
    }
    const filter = world.create();
    world.add(filter, FilterJoint, { a: bodies[6]!, b: bodies[8]! });
    joints.push(filter);
    return { bodies, joints };
}
