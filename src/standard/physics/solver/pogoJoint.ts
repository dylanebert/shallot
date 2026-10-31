import type { Vec3 } from "../common/records";
import { PGJ_IMPULSE, PGJ_VELOCITY } from "../kernel/columns";
import { readJointFloat } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** b3PogoJointDef, with resolved body handles. */
export type PogoJointDef = {
    base: JointDef;
    normal: Vec3;
    hertz: number;
    dampingRatio: number;
    restLength: number;
    maxTensionForce: number;
    maxCompressionForce: number;
    impulse: number;
    velocity: number;
};
export function defaultPogoJointDef(base: JointDef): PogoJointDef {
    return {
        base,
        normal: { x: 0, y: 1, z: 0 },
        hertz: 0,
        dampingRatio: 0,
        restLength: 0,
        maxTensionForce: 0,
        maxCompressionForce: 0,
        impulse: 0,
        velocity: 0,
    };
}
export function createPogoJoint(world: WorldState, def: PogoJointDef): { joint: Joint } {
    const joint = kernel(world.ecsState).jointCreatePogo(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.normal.x,
        def.normal.y,
        def.normal.z,
        def.restLength,
        def.hertz,
        def.dampingRatio,
        def.maxTensionForce,
        def.maxCompressionForce,
        def.impulse,
        def.velocity,
    );
    return finishJointCreation(world, def.base, joint);
}
export function pogoJointImpulse(world: WorldState, joint: Joint): number {
    return readJointFloat(world, joint, PGJ_IMPULSE);
}
export function pogoJointCurrentSpeed(world: WorldState, joint: Joint): number {
    return readJointFloat(world, joint, PGJ_VELOCITY);
}
export function pogoJointCurrentLength(world: WorldState, joint: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, joint, 10);
}
