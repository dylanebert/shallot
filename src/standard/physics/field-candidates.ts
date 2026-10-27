import type { World } from "../../engine";

export interface FieldCandidates {
    eids: number[];
    /** Queue slot plus one; zero means absent. */
    positions: Uint32Array;
    fieldMarks: Uint32Array;
    count: number;
    tombstones: number;
    /** The change tick of this queue's last capture; later captures latch only words marked since. */
    lastRun: number;
}

export interface FieldCandidateBinding {
    index: number;
    dirty: ReturnType<World["fieldStorage"]>[];
}

export function createFieldCandidates(): FieldCandidates {
    return {
        eids: [],
        positions: new Uint32Array(16),
        fieldMarks: new Uint32Array(16),
        count: 0,
        tombstones: 0,
        lastRun: 0,
    };
}

function ensureCandidateCapacity(candidates: FieldCandidates, eid: number): void {
    if (eid < candidates.positions.length) return;
    let capacity = candidates.positions.length;
    while (capacity <= eid) capacity *= 2;
    const positions = new Uint32Array(capacity);
    positions.set(candidates.positions);
    candidates.positions = positions;
    const fieldMarks = new Uint32Array(capacity);
    fieldMarks.set(candidates.fieldMarks);
    candidates.fieldMarks = fieldMarks;
}

export function clearFieldCandidate(candidates: FieldCandidates, eid: number): void {
    candidates.fieldMarks[eid] = 0;
}

/** Removes a queued candidate and its field marks, leaving a tombstone in its slot. */
export function dropFieldCandidate(candidates: FieldCandidates, eid: number): void {
    const position = candidates.positions[eid] ?? 0;
    if (position === 0) return;
    candidates.eids[position - 1] = 0;
    candidates.positions[eid] = 0;
    candidates.fieldMarks[eid] = 0;
    candidates.tombstones++;
}

export function markFieldCandidate(
    candidates: FieldCandidates,
    eid: number,
    fieldIndex?: number,
): void {
    ensureCandidateCapacity(candidates, eid);
    if (candidates.positions[eid] === 0) {
        candidates.eids[candidates.count++] = eid;
        candidates.positions[eid] = candidates.count;
    }
    // Components wider than one mask word fall back to comparing every field, never alias a low bit.
    if (fieldIndex !== undefined)
        candidates.fieldMarks[eid] |= fieldIndex < 32 ? 1 << fieldIndex : 0xffffffff;
}

/** Moves a candidate to the queue's end without losing field marks; zero slots are tombstones. */
export function requeueFieldCandidate(candidates: FieldCandidates, eid: number): void {
    const position = candidates.positions[eid] ?? 0;
    if (position !== 0) {
        if (position === candidates.count) return;
        candidates.eids[position - 1] = 0;
        candidates.positions[eid] = 0;
        candidates.tombstones++;
    }
    markFieldCandidate(candidates, eid);
    // Charge each stable compaction to the tombstones accumulated since the last one.
    if (candidates.tombstones > candidates.count - candidates.tombstones) {
        let live = 0;
        for (let i = 0; i < candidates.count; i++) {
            const queued = candidates.eids[i];
            if (queued === 0) continue;
            candidates.eids[live++] = queued;
            candidates.positions[queued] = live;
        }
        candidates.count = live;
        candidates.tombstones = 0;
    }
}

export function captureFieldCandidate(
    world: World,
    candidates: FieldCandidates,
    binding: FieldCandidateBinding,
): void {
    const words = (world.entityHighWater + 31) >>> 5;
    const since = candidates.lastRun;
    candidates.lastRun = world.incrementChangeTick();
    for (let fieldIndex = 0; fieldIndex < binding.dirty.length; fieldIndex++) {
        const dirty = binding.dirty[fieldIndex].dirty;
        const ticks = binding.dirty[fieldIndex].changeTicks;
        for (let word = 0; word < Math.min(dirty.length, words); word++) {
            if (ticks[word] <= since) continue;
            let bits = dirty[word];
            while (bits !== 0) {
                const low = bits & -bits;
                bits ^= low;
                const eid = (word << 5) | (31 - Math.clz32(low));
                markFieldCandidate(candidates, eid, fieldIndex);
            }
        }
    }
}

export function captureFieldCandidates(
    world: World,
    candidates: readonly FieldCandidates[],
    bindings: readonly FieldCandidateBinding[],
): void {
    for (let bindingIndex = 0; bindingIndex < bindings.length; bindingIndex++) {
        const binding = bindings[bindingIndex];
        const target = candidates[binding.index];
        if (target) captureFieldCandidate(world, target, binding);
    }
}

export function clearFieldCandidates(candidates: FieldCandidates): void {
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        candidates.positions[eid] = 0;
        candidates.fieldMarks[eid] = 0;
    }
    candidates.count = 0;
    candidates.tombstones = 0;
}
