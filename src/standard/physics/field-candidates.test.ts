import { expect, test } from "bun:test";
import { World } from "../../engine";
import { fieldMarked, shapeBinding } from "./authoring";
import {
    clearFieldCandidates,
    createFieldCandidates,
    markFieldCandidate,
    requeueFieldCandidate,
} from "./field-candidates";

test("requeue appends in consumption order without losing marks or duplicating live candidates", () => {
    const candidates = createFieldCandidates();
    markFieldCandidate(candidates, 1, 2);
    markFieldCandidate(candidates, 100, 4);
    markFieldCandidate(candidates, 3);
    requeueFieldCandidate(candidates, 100);
    requeueFieldCandidate(candidates, 1);
    expect(candidates.eids.slice(0, candidates.count)).toEqual([0, 0, 3, 100, 1]);
    expect(candidates.positions[100]).toBe(4);
    expect(candidates.positions[1]).toBe(5);
    expect(candidates.fieldMarks[100]).toBe(1 << 4);
    expect(candidates.fieldMarks[1]).toBe(1 << 2);
    requeueFieldCandidate(candidates, 1);
    markFieldCandidate(candidates, 100, 5);
    expect(candidates.count).toBe(5);
    expect(candidates.fieldMarks[100]).toBe((1 << 4) | (1 << 5));
    clearFieldCandidates(candidates);
    expect(candidates.positions[100]).toBe(0);
    expect(candidates.positions[1]).toBe(0);
    requeueFieldCandidate(candidates, 100);
    expect(candidates.eids.slice(0, candidates.count)).toEqual([100]);
});

test("100000 alternating requeues retain two live candidates in bounded slots without replacing storage", () => {
    const candidates = createFieldCandidates();
    markFieldCandidate(candidates, 1, 2);
    markFieldCandidate(candidates, 2, 4);
    const eids = candidates.eids,
        positions = candidates.positions,
        marks = candidates.fieldMarks;
    let maximum = 0;
    for (let i = 0; i < 100000; i++) {
        requeueFieldCandidate(candidates, (i % 2) + 1);
        maximum = Math.max(maximum, candidates.count);
    }
    expect(maximum).toBeLessThanOrEqual(4);
    expect(candidates.eids.length).toBeLessThanOrEqual(5);
    expect(candidates.eids.slice(0, candidates.count).filter((eid) => eid !== 0)).toEqual([1, 2]);
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        if (eid !== 0) expect(candidates.positions[eid]).toBe(i + 1);
    }
    expect(candidates.fieldMarks[1]).toBe(1 << 2);
    expect(candidates.fieldMarks[2]).toBe(1 << 4);
    expect(candidates.eids).toBe(eids);
    expect(candidates.positions).toBe(positions);
    expect(candidates.fieldMarks).toBe(marks);
    clearFieldCandidates(candidates);
    markFieldCandidate(candidates, 2);
    markFieldCandidate(candidates, 1);
    requeueFieldCandidate(candidates, 2);
    expect(candidates.eids.slice(0, candidates.count).filter((eid) => eid !== 0)).toEqual([1, 2]);
    requeueFieldCandidate(candidates, 1);
    requeueFieldCandidate(candidates, 2);
    expect(candidates.eids.slice(0, candidates.count)).toEqual([1, 2]);
    expect(candidates.positions[1]).toBe(1);
    expect(candidates.positions[2]).toBe(2);
    expect(candidates.fieldMarks[1]).toBe(0);
    expect(candidates.fieldMarks[2]).toBe(0);
});

test("marks above one word compare every field without aliasing a low field", () => {
    const world = new World();
    const binding = shapeBinding(world);
    const candidates = createFieldCandidates();
    const high = binding.fieldIndices.get("materialSet")!;
    expect(high).toBeGreaterThanOrEqual(32);
    markFieldCandidate(candidates, 1, 0);
    expect(fieldMarked(binding, candidates.fieldMarks[1]!, "body")).toBe(true);
    expect(fieldMarked(binding, candidates.fieldMarks[1]!, "materialSet")).toBe(false);
    markFieldCandidate(candidates, 1, high);
    expect(candidates.count).toBe(1);
    for (const field of binding.fields)
        expect(fieldMarked(binding, candidates.fieldMarks[1]!, field.name)).toBe(true);
    world.dispose();
});
