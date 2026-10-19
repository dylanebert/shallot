import type { World } from "../../engine";
import type { Quat, Vec3 } from "./common/records";
import { kernel } from "./kernel/kernel";

/** Box3D's deterministic axis-angle rotation, in radians about a normalized axis.
 * Requires an initialized StandardPhysicsPlugin. The returned quaternion belongs to the caller. */
export function rotationFromAxisAngle(world: World, axis: Vec3, radians: number): Quat {
    const k = kernel(world);
    new Float32Array(k.memory.buffer, k.mathInputPtr(), 4).set([axis.x, axis.y, axis.z, radians]);
    const q = new Float32Array(k.memory.buffer, k.mathRead(6), 4);
    return { v: { x: q[0]!, y: q[1]!, z: q[2]! }, s: q[3]! };
}
