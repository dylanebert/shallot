import type { State } from "../../src/engine";
import { Time } from "../../src/engine/ecs/scheduler";
import { WorldTransform } from "../../src/standard/transforms";

export function getWorldPosition(eid: number): { x: number; y: number; z: number } {
    const o = eid * 16;
    return {
        x: WorldTransform.data[o + 12],
        y: WorldTransform.data[o + 13],
        z: WorldTransform.data[o + 14],
    };
}

export function getWorldScale(eid: number): { x: number; y: number; z: number } {
    const d = WorldTransform.data;
    const o = eid * 16;
    const m00 = d[o],
        m01 = d[o + 1],
        m02 = d[o + 2];
    const m10 = d[o + 4],
        m11 = d[o + 5],
        m12 = d[o + 6];
    const m20 = d[o + 8],
        m21 = d[o + 9],
        m22 = d[o + 10];

    return {
        x: Math.sqrt(m00 * m00 + m01 * m01 + m02 * m02),
        y: Math.sqrt(m10 * m10 + m11 * m11 + m12 * m12),
        z: Math.sqrt(m20 * m20 + m21 * m21 + m22 * m22),
    };
}

export function count(state: State, components: unknown[]): number {
    let n = 0;
    for (const _ of state.query(components as never)) n++;
    return n;
}

export function first(state: State, components: unknown[]): number {
    for (const eid of state.query(components as never)) return eid;
    return -1;
}

export function all(state: State, components: unknown[]): number[] {
    return [...state.query(components as never)];
}

export function stepFor(state: State, duration: number): void {
    const maxDt = Time.FIXED_DT * Time.MAX_FIXED_STEPS;
    while (duration > maxDt) {
        state.step(maxDt);
        duration -= maxDt;
    }
    if (duration > 0) state.step(duration);
}

export function spawn(state: State, ...args: [unknown, Record<string, number>?][]): number {
    const eid = state.addEntity();
    for (const [component, values] of args) {
        state.addComponent(eid, component as never);
        if (values) {
            const comp = component as Record<string, number[]>;
            for (const [field, value] of Object.entries(values)) {
                if (field in comp) comp[field][eid] = value;
            }
        }
    }
    return eid;
}
