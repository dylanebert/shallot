// S3 arm — scripts/install-test.ts free-port probe (not hardcoded port)
//
// Invariant: the dev server port is picked by a free-port probe (createServer with port 0),
// not hardcoded to 5191. Before the S1 fix, port 5191 was hardcoded with no free-port probe —
// a collision reds the install gate as a product failure (false red). The fix added a freePort
// function that listens on port 0 and uses the OS-assigned port.
//
// install-test.ts needs a real `bun pm pack` + `bun install` (not hermetic), so this arm reads
// the source and asserts the free-port probe is present (structural pin).
//
// THIS SITE COUNTS AS UNARMED. A structural pin matches a commented-out guard as readily as a live
// one — measured: commenting the guard out leaves this arm GREEN. A source-text match cannot tell a
// guard from a comment, which is this spec's own defect class, so this file is a note and not
// coverage. Arming it behaviorally needs a hermetic pack-and-install fixture; that cost was not paid
// here.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const src = readFileSync(resolve(import.meta.dir, "install-test.ts"), "utf8");

test("install-test — uses a free-port probe (createServer with port 0), not a hardcoded port", () => {
    // The fix: a freePort function that listens on port 0 and uses the OS-assigned port.
    // The structural pin: the source contains a createServer call that listens on port 0.
    expect(src).toMatch(/createServer/);
    expect(src).toMatch(/\.listen\(0/);
    // The hardcoded port 5191 must not appear as a listen port.
    // (It may appear in comments describing the old behavior, but not as a listen argument.)
    expect(src).not.toMatch(/\.listen\(5191/);
});

test("install-test — the free-port probe is used for the dev server port", () => {
    // The fix: the dev server uses the free port, not a hardcoded one.
    // The freePort function is called to get the port for the dev server.
    expect(src).toMatch(/freePort/);
    // The port variable used for the dev server comes from freePort, not a literal.
    expect(src).toMatch(/await\s+freePort\(\)/);
});
