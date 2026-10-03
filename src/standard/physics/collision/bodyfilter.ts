import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";

/** Resident sorted unordered pairs, with reference counts for parallel non-colliding joints. */
export type BodyFilters = { data: Uint32Array; capacity: number };

function lowerBound(data: Uint32Array, a: number, b: number): number {
    let lo = 0;
    let hi = data[0] ?? 0;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        const p = 1 + 3 * mid;
        if (data[p] < a || (data[p] === a && data[p + 1] < b)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

export function bodiesFiltered(world: WorldState, bodyA: number, bodyB: number): boolean {
    return kernel(world.ecsState).broadBodiesFiltered(bodyA, bodyB) !== 0;
}

/** Called only on joint creation, destruction or a collideConnected transition. */
export function changeBodyFilter(
    world: WorldState,
    bodyA: number,
    bodyB: number,
    delta: 1 | -1,
): void {
    const filter = world.bodyFilters;
    const store = world.broadPhase.store;
    store.refreshIfStale();
    let data = filter.data;
    const a = Math.min(bodyA, bodyB);
    const b = Math.max(bodyA, bodyB);
    const i = lowerBound(data, a, b);
    const p = 1 + 3 * i;
    const count = data[0];
    if (i < count && data[p] === a && data[p + 1] === b) {
        data[p + 2] += delta;
        if (data[p + 2] === 0) {
            data.copyWithin(p, p + 3, 1 + 3 * count);
            data[0] = count - 1;
        }
        return;
    }
    if (delta < 0) throw new Error("physics: missing body filter");
    if (count === filter.capacity) {
        filter.capacity *= 2;
        store.growBodyFilters(filter.capacity);
        data = filter.data;
    }
    data.copyWithin(p + 3, p, 1 + 3 * count);
    data[p] = a;
    data[p + 1] = b;
    data[p + 2] = 1;
    data[0] = count + 1;
}
