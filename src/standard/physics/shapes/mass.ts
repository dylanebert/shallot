import type { World } from "../../../engine";
import { kernel } from "../kernel/kernel";
import type { Capsule, MassData } from "./geometry";
import type { HullData } from "./hull";

function input(world?: World): Float32Array {
    const k = kernel(world);
    const ptr = k.hullBuildInputPtr(2);
    return new Float32Array(k.memory.buffer, ptr, 16);
}
function result(operation: number, world?: World): MassData {
    const k = kernel(world);
    k.geometryMass(operation);
    const r = new Float32Array(k.memory.buffer, k.hullBuildInputPtr(2), 13);
    return {
        mass: r[0],
        center: { x: r[1], y: r[2], z: r[3] },
        inertia: {
            cx: { x: r[4], y: r[5], z: r[6] },
            cy: { x: r[7], y: r[8], z: r[9] },
            cz: { x: r[10], y: r[11], z: r[12] },
        },
    };
}
/**
 * Owned mass in kg, local center in meters and inertia about that center in kg·m².
 * Density is nonnegative kg/m³; capsule dimensions are meters. A zero-length segment
 * gives sphere mass. Requires an initialized StandardPhysicsPlugin; the optional World
 * selects its kernel. Inputs are not retained; each construction-time call allocates a result.
 */
export function computeCapsuleMass(capsule: Capsule, density: number, world?: World): MassData {
    if (!Number.isFinite(density) || density < 0) throw new RangeError("Invalid density");
    const r = input(world);
    r[0] = capsule.center1.x;
    r[1] = capsule.center1.y;
    r[2] = capsule.center1.z;
    r[3] = capsule.center2.x;
    r[4] = capsule.center2.y;
    r[5] = capsule.center2.z;
    r[6] = capsule.radius;
    r[7] = density;
    return result(0, world);
}
/**
 * Owned mass in kg, local center in meters and inertia about that center in kg·m²,
 * from a built hull's bulk properties and nonnegative density in kg/m³.
 * Requires an initialized StandardPhysicsPlugin; the optional World selects its kernel.
 * The hull is not retained; each construction-time call allocates a result.
 */
export function computeHullMass(hull: HullData, density: number, world?: World): MassData {
    if (!Number.isFinite(density) || density < 0) throw new RangeError("Invalid density");
    const r = input(world);
    r[0] = density;
    r[1] = hull.volume;
    r[2] = hull.center.x;
    r[3] = hull.center.y;
    r[4] = hull.center.z;
    const i = hull.centralInertia;
    r[5] = i.cx.x;
    r[6] = i.cx.y;
    r[7] = i.cx.z;
    r[8] = i.cy.x;
    r[9] = i.cy.y;
    r[10] = i.cy.z;
    r[11] = i.cz.x;
    r[12] = i.cz.y;
    r[13] = i.cz.z;
    return result(1, world);
}
