// Fixture for `packages/shallot/tests/test-cap.test.ts`: a file the cap must red when the cap is
// driven to 0 ms. Named `.fixture.ts` so bun's default test-file pattern does not collect it under
// `bun test` — and `bunfig.toml`'s `root = "."` (widened from `"packages/shallot"` in S3) scopes
// discovery to the repo root, so a fixture outside `packages/shallot` is collected by a bare
// `bun test` — but named `.fixture.ts` so the default test-file pattern does not pick it up.
// An explicitly passed path still runs even when it does not match the pattern, which is how
// the arms invoke it.
//
// Two tests, so the once-per-file latch in the preload is exercised: exactly one red.
import { expect, test } from "bun:test";

test("over-cap fixture arm 1", () => {
    expect(1).toBe(1);
});

test("over-cap fixture arm 2", () => {
    expect(2).toBe(2);
});
