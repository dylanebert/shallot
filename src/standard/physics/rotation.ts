import type { World } from "../../engine";
import type { Vec3 } from "./common/records";
import { kernel } from "./kernel/kernel";

/** Box3D's deterministic axis-angle rotation, in radians about a normalized axis, as an xyzw quaternion.
 * Requires an initialized StandardPhysicsPlugin. The returned tuple belongs to the caller. */
export function rotationFromAxisAngle(
    world: World,
    axis: Vec3,
    radians: number,
): [number, number, number, number] {
    const k = kernel(world);
    new Float32Array(k.memory.buffer, k.mathInputPtr(), 4).set([axis.x, axis.y, axis.z, radians]);
    const q = new Float32Array(k.memory.buffer, k.mathRead(6), 4);
    return [q[0]!, q[1]!, q[2]!, q[3]!];
}
