// S3 arm — scripts/physics-bench.ts empty derived population reads red
//
// Invariant: an empty derived population (scenarioRows) makes scenariosGreen false (red), not
// vacuously true (green). Before the S1 fix, scenariosGreen used a bare .every() on scenarioRows
// — an empty array's .every() returns true (vacuous truth), so an empty sweep read green.
// The fix added `scenarioRows.length > 0 &&` ahead of the .every().
//
// physics-bench.ts is a GPU-gated top-level script (can't run hermetically), so the arm extracts
// the scenariosGreen expression from the source and evaluates it with an empty array — a
// behavioral test of the actual expression in the script, not a copy. If the guard is removed,
// the expression evaluates to true (vacuous green) and the arm reds.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const src = readFileSync(resolve(import.meta.dir, "physics-bench.ts"), "utf8");

test("physics-bench — scenariosGreen is false for an empty population (the empty-guard)", () => {
    // Extract the scenariosGreen expression from the source and evaluate it with an empty array.
    // The expression is: scenarioRows.length > 0 && scenarioRows.every(...)
    // With an empty array: false (the guard fires). Without the guard: true (vacuous green).
    const match = src.match(/const\s+scenariosGreen\s*=\s*([\s\S]*?);/);
    expect(match, "scenariosGreen expression found in source").not.toBeNull();
    const expr = match![1];

    // Evaluate the expression with an empty scenarioRows array.
    // The expression references `scenarioRows` and `r.name` / `r.error` / `r.step` on its elements.
    const fn = new Function("scenarioRows", `return ${expr};`) as (rows: unknown[]) => boolean;

    // With the guard: empty array → false (red). This is the invariant.
    expect(fn([])).toBe(false);

    // Without the guard: empty array → true (vacuous green). This is what the fix prevents.
    // Verify the semantics: [].every(...) returns true, but [].length > 0 && [].every(...) is false.
    const empty: { error?: string; step: number; name: string }[] = [];
    expect(empty.every((r) => !r.error && (r.step > 0 || r.name.includes("probe")))).toBe(true);
    expect(
        empty.length > 0 &&
            empty.every((r) => !r.error && (r.step > 0 || r.name.includes("probe"))),
    ).toBe(false);
});

test("physics-bench — the guard is `length > 0 &&`, not just .every()", () => {
    // Structural pin: the scenariosGreen expression must contain the length guard.
    // This reds if someone removes the `length > 0 &&` and leaves only the bare .every().
    const match = src.match(/const\s+scenariosGreen\s*=\s*([\s\S]*?);/);
    expect(match).not.toBeNull();
    const expr = match![1];
    expect(expr).toMatch(/\.length\s*>\s*0\s*&&/);
});
