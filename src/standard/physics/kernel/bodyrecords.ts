import type { Mat3 } from "../common/math";
import type { WorldState } from "../world/world";

export const BODY_RECORD_STRIDE = 29;
export const BodyField = {
    setIndex: 0,
    localIndex: 1,
    headContactKey: 2,
    contactCount: 3,
    headShapeId: 4,
    shapeCount: 5,
    headChainId: 6,
    headJointKey: 7,
    jointCount: 8,
    islandId: 9,
    islandIndex: 10,
    sleepThreshold: 11,
    sleepTime: 12,
    sleepVelocity: 13,
    mass: 14,
    bodyMoveIndex: 24,
    id: 25,
    flags: 26,
    type: 27,
    generation: 28,
    userData: 29,
    name: 30,
} as const;
export type BodyField = (typeof BodyField)[keyof typeof BodyField];

export function bodyField(world: WorldState, id: number, field: typeof BodyField.name): string;
export function bodyField(world: WorldState, id: number, field: typeof BodyField.userData): unknown;
export function bodyField(world: WorldState, id: number, field: BodyField): number;
export function bodyField(world: WorldState, id: number, field: BodyField): unknown {
    if (field === BodyField.userData) return world.bodyUserData[id];
    if (field === BodyField.name) return world.bodyNames[id] ?? "";
    const store = world.bodyStore;
    const offset = id * BODY_RECORD_STRIDE + field;
    if (field >= BodyField.sleepThreshold && field <= BodyField.mass) return store.recordF[offset];
    if (field === BodyField.generation) return store.recordU[offset] & 0xffff;
    if (field === BodyField.flags) return store.recordU[offset];
    return store.recordU[offset] | 0;
}

export function setBodyField(
    world: WorldState,
    id: number,
    field: BodyField,
    value: unknown,
): void {
    if (field === BodyField.userData) {
        world.bodyUserData[id] = value;
        return;
    }
    if (field === BodyField.name) {
        world.bodyNames[id] = value as string;
        return;
    }
    const offset = id * BODY_RECORD_STRIDE + field;
    if (field >= BodyField.sleepThreshold && field <= BodyField.mass)
        world.bodyStore.recordF[offset] = value as number;
    else
        world.bodyStore.recordU[offset] =
            field === BodyField.generation ? (value as number) & 0xffff : (value as number);
}

export function bodyInertia(world: WorldState, id: number, out: Mat3): Mat3 {
    const f = world.bodyStore.recordF;
    const o = id * BODY_RECORD_STRIDE + 15;
    out.cx.x = f[o];
    out.cx.y = f[o + 1];
    out.cx.z = f[o + 2];
    out.cy.x = f[o + 3];
    out.cy.y = f[o + 4];
    out.cy.z = f[o + 5];
    out.cz.x = f[o + 6];
    out.cz.y = f[o + 7];
    out.cz.z = f[o + 8];
    return out;
}
