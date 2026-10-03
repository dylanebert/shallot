import { FLT_MAX, f32, type Quat, quat, type Vec2, type Vec3, vec3 } from "../common/math";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Parallel joint payload (b3ParallelJoint). Impulse persists across steps for warm starting. */
export type ParallelJoint = {
    perpImpulse: Vec2;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
    quatA: Quat;
    quatB: Quat;
    perpAxisX: Vec3;
    perpAxisY: Vec3;
};

/** Parallel joint definition (b3ParallelJointDef), body handles resolved to a base JointDef. */
export type ParallelJointDef = {
    base: JointDef;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** @returns the ported parallel joint definition defaults (b3DefaultParallelJointDef). */
export function defaultParallelJointDef(base: JointDef): ParallelJointDef {
    return { base, hertz: 1, dampingRatio: 1, maxTorque: FLT_MAX };
}

const identityQuat = (): Quat => ({ v: { x: 0, y: 0, z: 0 }, s: 1 });

/** Create a parallel joint (b3CreateParallelJoint). @returns the joint handle + sim. */
export function createParallelJoint(
    world: WorldState,
    def: ParallelJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Parallel);
    const data: ParallelJoint = {
        perpImpulse: { x: 0, y: 0 },
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxTorque: def.maxTorque,
        quatA: identityQuat(),
        quatB: identityQuat(),
        perpAxisX: { x: 0, y: 0, z: 0 },
        perpAxisY: { x: 0, y: 0, z: 0 },
    };
    pair.sim.data = data;
    return pair;
}

// The two perpendicular collinearity axes in world space, from the relative rotation (relQ) of the
// two joint frames. relQ = inv(quatA) * quatB; the axes are half the rotated imaginary parts.
function perpAxes(qA: Quat, relQ: Quat): { x: Vec3; y: Vec3 } {
    return {
        x: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
            ),
        ),
        y: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
            ),
        ),
    };
}

export function getParallelJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as ParallelJoint;
    const relQ = quat.invMul(joint.quatA, joint.quatB);
    const axes = perpAxes(joint.quatA, relQ);
    joint.perpAxisX = axes.x;
    joint.perpAxisY = axes.y;

    const angularImpulse = vec3.blend2(
        joint.perpImpulse.x,
        joint.perpAxisX,
        joint.perpImpulse.y,
        joint.perpAxisY,
    );
    return vec3.scale(world.invH, angularImpulse);
}
