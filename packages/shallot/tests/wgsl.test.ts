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
     * A signature present in the source but with no `{` after it: `body()` must throw
     * `unterminated body`, not return a slice. Before the guard, `src.indexOf("{", start)` was
     * `-1`, so the loop started at `i = -1` (where `src[-1]` is `undefined`, matching neither
     * branch), then rescanned from index 0 — finding the `{` *before* the signature and returning
     * `src.slice(start, i + 1)` where `i + 1 < start`, i.e. an **empty string** — instead of
     * throwing. An empty slice satisfies every `not.toMatch` in `noIntegerDivision`, so a
     * discipline gate passed by extracting nothing.
     */
    test("a signature present but brace-less must throw", () => {
        const src = "{ x } fn baz() no body here";
        expect(() => body(src, "fn baz()")).toThrow("unterminated body");
    });

    /**
     * A normal signature with a balanced body: `body()` must return the full slice from the
     * signature through the matching close brace. Before the guard this arm was green (the
     * defect only affected the brace-less path), and it stays green after — so it pins that the
     * guard's condition is exactly "no brace after the signature" and not something broader that
     * would throw on valid input. An input that reds this arm is one where a `{` follows the
     * signature but the guard throws anyway.
     */
    test("a normal signature returns the balanced slice", () => {
        const src = "fn baz() { return 0; }";
        expect(body(src, "fn baz()")).toBe("fn baz() { return 0; }");
    });
});
