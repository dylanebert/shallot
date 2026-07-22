/**
 * the CPU pre-pack stamp diff (ecs.md "An eid is a borrow") — given the live `Body` eids
 * and their create-stamps, return the eids RECYCLED to a new body since the last pack, updating `seen` in
 * place. A same-update destroy+create realias keeps an eid a `Body` member the whole time, so the GPU
 * pack's non-member seed reset never fires on it; the create-stamp is the only signal that the slot now
 * holds a new body. A first-seen eid is a fresh spawn (the GPU pack seeds it via its own gate) — not
 * recycled; only a stamp that changed against a known prior one is a realias. The caller reseeds each
 * returned eid (clear the GPU seed flag + evict the kinematic-prev pose).
 */
export function diffStamps(
    eids: Iterable<number>,
    stampOf: (eid: number) => number,
    seen: Map<number, number>,
): number[] {
    const recycled: number[] = [];
    for (const eid of eids) {
        const stamp = stampOf(eid);
        if (seen.get(eid) !== stamp) {
            if (seen.has(eid)) recycled.push(eid);
            seen.set(eid, stamp);
        }
    }
    return recycled;
}
