import { describe, expect, test } from "bun:test";
import { parseArgs } from "../../../scripts/bench";

// HOLE-RECORD ARMS — `scripts/bench.ts`'s six numeric flags (`--seed`, `--count`, `--warmup`, `--frames`,
// `--timeout`, `--leak`) are each `parseInt`'d without validation (bench.ts:240-261), so a non-numeric
// value silently becomes NaN and flows into the query string (`warmup=NaN`) or `--timeout NaN`. verify.ts:112's
// own JSDoc states the policy — "a typo must not silently no-op or flow NaN" — the same repo's stated rule,
// unapplied one file over.
//
// Each bare `toThrow()` below distinguishes a throw from a clean return, so a NaN-substituting default
// leaves it red — but it CANNOT distinguish the validation throw from an unrelated one (an unknown-option
// guard, a precondition firing first), so a message assertion is owed the moment a message exists. S2 of
// this spec (audit-cli-numeric-flags) is the unit that flips these arms green, and it tightens each
// `toThrow()` to name the flag in the message rather than leaving the fragment loose.

describe("parseArgs — bench.ts numeric flag validation (RED today)", () => {
    test("--count abc throws instead of flowing NaN into the query string — RED today", () => {
        expect(() => parseArgs(["--count", "abc"])).toThrow();
    });

    test("--seed abc throws instead of flowing NaN — RED today", () => {
        expect(() => parseArgs(["--seed", "abc"])).toThrow();
    });

    test("--warmup abc throws instead of flowing warmup=NaN into the query string — RED today", () => {
        expect(() => parseArgs(["--warmup", "abc"])).toThrow();
    });

    test("--frames abc throws instead of flowing NaN — RED today", () => {
        expect(() => parseArgs(["--frames", "abc"])).toThrow();
    });

    test("--timeout abc throws instead of flowing --timeout NaN — RED today", () => {
        expect(() => parseArgs(["--timeout", "abc"])).toThrow();
    });

    test("--leak abc throws instead of flowing NaN — RED today", () => {
        expect(() => parseArgs(["--leak", "abc"])).toThrow();
    });
});
