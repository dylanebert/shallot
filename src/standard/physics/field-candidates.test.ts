import { expect, test } from "bun:test";
import { World } from "../../engine";
import { fieldMarked, shapeBinding } from "./authoring";
import { createFieldCandidates, markFieldCandidate } from "./field-candidates";

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
