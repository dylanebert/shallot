import { BodyType } from "../common/types";
import { kernel } from "../kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import { bodySetTransform } from "../world/body";
import {
    createJoint,
    defaultJointDef,
    destroyJointInternal,
    getJointAngularSeparation,
    getJointLinearSeparation,
    type JointType,
} from "./joint";
import { revoluteJointAngle } from "./revoluteJoint";
import { sphericalJointConeAngle, sphericalJointTwistAngle } from "./sphericalJoint";
import { wheelJointSteeringAngle } from "./wheelJoint";
export type JointReadCase = {
    kind: number;
    sample: number;
    a: number[];
    b: number[];
    joint: number[];
    output: (number | null)[];
};
export function visitJointReaders(
    cases: JointReadCase[],
    check: (actual: number, expected: number, label: string) => void,
): void {
    const world = createTestWorld();
    const a = createTestBody(world, { type: BodyType.Dynamic }),
        b = createTestBody(world, { type: BodyType.Dynamic });
    const pose = (bits: number[]) => {
        const f = new Float32Array(new Uint32Array(bits).buffer);
        return {
            p: { x: f[0], y: f[1], z: f[2] },
            q: { v: { x: f[3], y: f[4], z: f[5] }, s: f[6] },
        };
    };
    const getters = [
        revoluteJointAngle,
        sphericalJointConeAngle,
        sphericalJointTwistAngle,
        wheelJointSteeringAngle,
        getJointAngularSeparation,
        getJointLinearSeparation,
    ];
    try {
        for (const c of cases) {
            const pa = pose(c.a),
                pb = pose(c.b);
            bodySetTransform(world, a, pa.p, pa.q);
            bodySetTransform(world, b, pb.p, pb.q);
            const j = createJoint(
                world,
                { ...defaultJointDef(), bodyIdA: a, bodyIdB: b },
                c.kind as JointType,
            ).joint;
            const k = kernel(world.ecsState);
            k.bodySetActiveWorld(world.worldId);
            new Uint32Array(k.memory.buffer, k.jointSimPtr(j), c.joint.length).set(c.joint);
            try {
                for (let op = 0; op < c.output.length; ++op) {
                    const expected = c.output[op];
                    if (expected === null) continue;
                    const actual = new Uint32Array(
                        new Float32Array([getters[op](world, j)]).buffer,
                    )[0];
                    check(actual, expected, `kind ${c.kind}, sample ${c.sample}, reader ${op}`);
                }
            } finally {
                destroyJointInternal(world, j, false);
            }
        }
    } finally {
        destroyTestWorld(world);
    }
}
