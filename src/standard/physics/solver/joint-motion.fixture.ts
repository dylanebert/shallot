import { BodyType, defaultShapeDef } from "../common/types";
import { kernel } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import { bodySetAngularVelocity, bodySetAwake, bodySetLinearVelocity } from "../world/body";
import { distanceJointCurrentLength } from "./distanceJoint";
import { createJoint, defaultJointDef, type JointType } from "./joint";
import { prismaticJointSpeed, prismaticJointTranslation } from "./prismaticJoint";
import { wheelJointSpinSpeed } from "./wheelJoint";
export type JointMotionCase = {
    kind: number;
    mode: number;
    sample: number;
    a: number[];
    b: number[];
    joint: number[];
    output: (number | null)[];
};
export function visitJointMotion(
    cases: JointMotionCase[],
    check: (actual: number, expected: number, label: string) => void,
): void {
    const getters = [
        prismaticJointTranslation,
        distanceJointCurrentLength,
        prismaticJointSpeed,
        wheelJointSpinSpeed,
    ];
    for (const c of cases) {
        const world = createTestWorld();
        const motion = (bits: number[]) => {
            const f = new Float32Array(new Uint32Array(bits).buffer);
            const v = (i: number) => ({ x: f[i], y: f[i + 1], z: f[i + 2] });
            return {
                position: v(0),
                rotation: { v: v(3), s: f[6] },
                linear: v(13),
                angular: v(16),
            };
        };
        try {
            const ma = motion(c.a),
                mb = motion(c.b);
            const a = createTestBody(world, {
                type:
                    c.mode === 1
                        ? BodyType.Static
                        : c.mode === 3
                          ? BodyType.Kinematic
                          : BodyType.Dynamic,
                position: ma.position,
                rotation: ma.rotation,
            });
            const b = createTestBody(world, {
                type:
                    c.mode === 2
                        ? BodyType.Static
                        : c.mode === 4
                          ? BodyType.Kinematic
                          : BodyType.Dynamic,
                position: mb.position,
                rotation: mb.rotation,
            });
            createSphereShape(world, a, defaultShapeDef(), {
                center: { x: 0.3, y: -0.2, z: 0.1 },
                radius: 0.7,
            });
            createSphereShape(world, b, defaultShapeDef(), {
                center: { x: -0.4, y: 0.5, z: -0.6 },
                radius: 0.7,
            });
            bodySetLinearVelocity(world, a, ma.linear);
            bodySetAngularVelocity(world, a, ma.angular);
            bodySetLinearVelocity(world, b, mb.linear);
            bodySetAngularVelocity(world, b, mb.angular);
            const j = createJoint(
                world,
                { ...defaultJointDef(), bodyIdA: a, bodyIdB: b },
                c.kind as JointType,
            ).joint;
            if (c.mode === 5) {
                bodySetAwake(world, a, false);
                bodySetAwake(world, b, false);
            }
            const k = kernel(world.ecsState);
            k.bodySetActiveWorld(world.worldId);
            new Uint32Array(k.memory.buffer, k.jointSimPtr(j), c.joint.length).set(c.joint);
            for (let op = 0; op < 4; ++op) {
                const expected = c.output[op];
                if (expected === null) continue;
                const actual = new Uint32Array(new Float32Array([getters[op](world, j)]).buffer)[0];
                check(
                    actual,
                    expected,
                    `reader ${op + 6}, kind ${c.kind}, mode ${c.mode}, sample ${c.sample}`,
                );
            }
        } finally {
            destroyTestWorld(world);
        }
    }
}
