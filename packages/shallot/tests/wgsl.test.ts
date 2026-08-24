/**
 * Arms for {@link body} — the brace-matched extraction helper in `wgsl.ts`. Two arms, per the spec's
 * Validation: one that a signature present but brace-less must throw (the defect: `body()` started
 * brace matching at `src.indexOf("{", start)`, which is `-1` when no brace follows the signature, so
 * the loop rescanned from index 0 and returned an arbitrary — possibly empty — slice instead of
 * reaching its own `unterminated body` throw), and one that a normal signature returns the balanced
 * slice (so the guard cannot be satisfied by making `body()` throw more often than it should).
 */
import { describe, expect, test } from "bun:test";
import { body } from "./wgsl";

describe("body", () => {
    /**
     * A signature present in the source but with no `{` after it: `body()` must throw from
     * the no-brace guard, not return a slice. The `toThrow("no opening brace after signature")`
     * matcher pins the guard's site specifically — the loop-end throw emits a different message
     * (`unterminated body for …`), so a regression that drops the guard and lets the loop
     * rescan reds this arm on the message, not just on "threw something." Before the guard,
     * `src.indexOf("{", start)` was `-1`, so the loop started at `i = -1` (where `src[-1]` is
     * `undefined`, matching neither branch), then rescanned from index 0 — finding the `{`
     * *before* the signature and returning `src.slice(start, i + 1)` where `i + 1 < start`,
     * i.e. an **empty string** — instead of throwing. An empty slice satisfies every
     * `not.toMatch` in `noIntegerDivision`, so a discipline gate fed this helper would pass by
     * extracting nothing. Latent, not historical: `body()` has no caller today (`standards.ts`
     * imports the four discipline helpers, not this one), which is the spec's "zero current
     * instances" — the guard exists so the first caller the TypeGPU port adds cannot inherit a
     * silent green.
     */
    test("a signature present but brace-less must throw", () => {
        const src = "{ x } fn baz() no body here";
        expect(() => body(src, "fn baz()")).toThrow("no opening brace after signature");
    });

    /**
     * A normal signature with a balanced body in a realistic multi-function module: `body()`
     * must return the full slice from the targeted signature through its matching close brace.
     * Every production caller passes a whole emitted WGSL module and slices one function out of
     * the middle, so the signature sits after a struct and a preceding function (`start > 0`
     * always), with a trailing function after — the equality on the returned slice can
     * distinguish that the slice is the body of the *targeted* function (a guard that found
     * the wrong `{` — from preceding content — and returned a different function's body would
     * red), but it cannot see *why* `body()` found the right brace, only that the slice
     * matches. The position (`start > 0`) is one of the things the fixture buys: at
     * `start = 0` a guard that searched from the start of the string instead of from the
     * signature cannot be caught (the wrong brace and the right brace are the same), but
     * once `start > 0` a preceding `{` sits before the signature, so the wrong-brace class
     * reds where it stayed green at index 0. Before the guard this arm was green (the defect
     * only affected the brace-less path), and it stays green after — so it pins that the
     * guard does not throw on valid input.
     */
    test("a normal signature returns the balanced slice", () => {
        const src = [
            "struct Vertex { pos: vec3<f32>, }",
            "fn other() -> vec3<f32> { return vec3<f32>(0.0, 0.0, 0.0); }",
            "fn baz() -> i32 { return 0; }",
            "fn after() -> u32 { return 1u; }",
        ].join("\n");
        expect(body(src, "fn baz()")).toBe("fn baz() -> i32 { return 0; }");
    });
});
