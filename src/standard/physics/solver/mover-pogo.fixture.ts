import { BodyType, defaultShapeDef } from "../common/types";
import { kernel } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld } from "../testing";
import { bodyGetLinearVelocity, bodyGetTransform } from "../world/body";
import {
    createJoint,
    defaultJointDef,
    destroyJointInternal,
    getJointConstraintForce,
    type JointType,
} from "./joint";

export type MoverPogoCase = { kind: number; mode: number; fields: number[][]; samples: number[][] };
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0]!;
export function visitMoverPogo(
    c: MoverPogoCase,
    check: (actual: number[], expected: number[], label: string) => void,
): void {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        const a = createTestBody(world);
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: c.mode === 1 ? 2 : 0.5, z: 0 },
            linearVelocity: { x: 4, y: c.mode === 1 ? 4 : -4, z: -3 },
        });
        createSphereShape(world, b, defaultShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const def = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
        const k = kernel(world.ecsState);
        let joint = createJoint(world, def, c.kind as JointType).joint;
        const words = () => {
            k.bodySetActiveWorld(world.worldId);
            return new Uint32Array(k.memory.buffer, k.jointSimPtr(joint), k.jointLayoutOffset(0));
        };
        const initialize = () => {
            const w = words();
            for (const [field, value] of c.fields) w[field!] = value!;
        };
        initialize();
        const pose = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            velocity = { x: 0, y: 0, z: 0 };
        for (let i = 0; i < c.samples.length; ++i) {
            if (i && c.mode === 2) {
                const old = words();
                const impulse = c.kind === 10 ? old[c.fields[8]![0]!]! : 0;
                const speed = c.kind === 10 ? old[c.fields[9]![0]!]! : 0;
                destroyJointInternal(world, joint, true);
                joint = createJoint(world, def, c.kind as JointType).joint;
                initialize();
                if (c.kind === 10) {
                    const w = words();
                    w[c.fields[8]![0]!] = impulse;
                    w[c.fields[9]![0]!] = speed;
                }
            }
            stepTestWorld(world, 1 / 60, 4);
            bodyGetTransform(world, b, pose);
            bodyGetLinearVelocity(world, b, velocity);
            const motion = [pose.p.x, pose.p.y, pose.p.z, velocity.x, velocity.y, velocity.z].map(
                bits,
            );
            check(
                motion,
                c.samples[i]!.slice(0, 6),
                `joint ${c.kind}, mode ${c.mode}, step ${i} motion`,
            );
            const force = getJointConstraintForce(world, joint);
            const actual = [...motion, ...[force.x, force.y, force.z].map(bits)];
            if (c.kind === 10) {
                const w = words();
                actual.push(w[c.fields[8]![0]!]!, w[c.fields[9]![0]!]!);
            }
            check(actual, c.samples[i]!, `joint ${c.kind}, mode ${c.mode}, step ${i}`);
        }
    } finally {
        destroyTestWorld(world);
    }
}
