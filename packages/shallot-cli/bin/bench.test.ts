import { describe, expect, test } from "bun:test";
import { parseArgs } from "../../../scripts/bench";

// `scripts/bench.ts`'s six numeric flags (`--seed`, `--count`, `--warmup`, `--frames`, `--timeout`,
// `--leak`) each reject a non-numeric value with a message naming the flag, so a typo doesn't flow NaN
// into the query string (`warmup=NaN`) or `--timeout NaN` (the policy in verify.ts's `--port` JSDoc:
// "a typo must not silently no-op or flow NaN"). Each message assertion distinguishes the validation
// throw from an unrelated one (an unknown-option guard, a precondition firing first).

describe("parseArgs — bench.ts numeric flag validation", () => {
    test("--count abc throws with a message naming the flag instead of flowing NaN into the query string", () => {
        expect(() => parseArgs(["--count", "abc"])).toThrow('invalid --count value "abc"');
    });

    test("--seed abc throws with a message naming the flag instead of flowing NaN", () => {
        expect(() => parseArgs(["--seed", "abc"])).toThrow('invalid --seed value "abc"');
    });

    test("--warmup abc throws with a message naming the flag instead of flowing warmup=NaN into the query string", () => {
        expect(() => parseArgs(["--warmup", "abc"])).toThrow('invalid --warmup value "abc"');
    });

    test("--frames abc throws with a message naming the flag instead of flowing NaN", () => {
        expect(() => parseArgs(["--frames", "abc"])).toThrow('invalid --frames value "abc"');
    });

    test("--timeout abc throws with a message naming the flag instead of flowing --timeout NaN", () => {
        expect(() => parseArgs(["--timeout", "abc"])).toThrow('invalid --timeout value "abc"');
    });

    test("--leak abc throws with a message naming the flag instead of flowing NaN", () => {
        expect(() => parseArgs(["--leak", "abc"])).toThrow('invalid --leak value "abc"');
    });

    test("an empty --seed value is rejected as empty, not silently coerced to 0", () => {
        expect(() => parseArgs(["--seed", ""])).toThrow(
            'invalid --seed value "" — must not be empty',
        );
    });

    test("a whitespace-only --seed value is rejected, not silently coerced to 0", () => {
        expect(() => parseArgs(["--seed", " "])).toThrow(
            'invalid --seed value " " — must not be empty',
        );
    });
});
