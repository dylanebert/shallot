// S3 arm — scripts/verify.ts discarded child exit code reddens the verdict
//
// Invariant: a discarded child exit code reddens the verdict. Before the S1 fix, spawnVerify
// awaited proc.exited and discarded the code — the verdict rested entirely on parsed stdout.
// The fix returns exitCode from spawnVerify, and verify() checks `exitCode !== 0` and sets
// pass: false regardless of what the parsed stdout says.
//
// The arm calls the exported verify() with a non-existent dir. The CLI subprocess fails (EXIT_SETUP,
// nonzero) and verify() must return pass: false. This is a behavioral test of the actual function.

import { expect, test } from "bun:test";
import { verify } from "./verify";

test("verify — a nonzero child exit reddens the verdict (pass: false)", async () => {
    // Call verify() with a non-existent dir. The CLI subprocess exits nonzero (EXIT_SETUP=2 or
    // EXIT_NO_PLAYWRIGHT=3), and verify() must return pass: false regardless of parsed stdout.
    const result = await verify("/nonexistent-shallot-arm-dir-12345", [], true);
    expect(result).not.toBeNull();
    expect(result!.pass).toBe(false);
});
