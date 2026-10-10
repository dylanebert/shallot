import type { World } from "../../engine";

export interface FieldCandidates {
    eids: number[];
    seen: Uint8Array;
    fieldMarks: Uint32Array;
    count: number;
}

export interface FieldCandidateBinding {
    index: number;
    dirty: ReturnType<World["fieldStorage"]>[];
}

export function createFieldCandidates(): FieldCandidates {
    return { eids: [], seen: new Uint8Array(16), fieldMarks: new Uint32Array(16), count: 0 };
}

function ensureCandidateCapacity(candidates: FieldCandidates, eid: number): void {
    if (eid < candidates.seen.length) return;
    let capacity = candidates.seen.length;
    while (capacity <= eid) capacity *= 2;
    const seen = new Uint8Array(capacity);
    seen.set(candidates.seen);
    candidates.seen = seen;
    const fieldMarks = new Uint32Array(capacity);
    fieldMarks.set(candidates.fieldMarks);
    candidates.fieldMarks = fieldMarks;
}

export function clearFieldCandidate(candidates: FieldCandidates, eid: number): void {
    candidates.fieldMarks[eid] = 0;
}

export function markFieldCandidate(
    candidates: FieldCandidates,
    eid: number,
    fieldIndex?: number,
): void {
    if (fieldIndex !== undefined && fieldIndex >= 32)
        throw new Error(`physics: field index ${fieldIndex} exceeds its candidate mask`);
    ensureCandidateCapacity(candidates, eid);
    if (candidates.seen[eid] === 0) {
        candidates.seen[eid] = 1;
        candidates.eids[candidates.count++] = eid;
    }
    if (fieldIndex !== undefined) candidates.fieldMarks[eid] |= 1 << fieldIndex;
}

export function captureFieldCandidate(
    world: World,
    candidates: FieldCandidates,
    binding: FieldCandidateBinding,
): void {
    const words = (world.entityHighWater + 31) >>> 5;
    for (let fieldIndex = 0; fieldIndex < binding.dirty.length; fieldIndex++) {
        const dirty = binding.dirty[fieldIndex].dirty;
        for (let word = 0; word < Math.min(dirty.length, words); word++) {
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
        candidates.seen[eid] = 0;
        candidates.fieldMarks[eid] = 0;
    }
    candidates.count = 0;
}
