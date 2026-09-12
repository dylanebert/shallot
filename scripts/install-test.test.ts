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
//
// The costly packed-install admission runs by path in scripts/install-test.probes.ts.

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

// The previous release's built-verify leg attaches to a browser server; the candidate's does not. The
// candidate's own public headless launch is part of what this flow verifies, so asserting only the
// `previous` shape would pass just as well if both carried the flag.

import { missingCrateDiagnosticPass } from "./install-test";
import { readEndpoint, verifyArgs } from "./install-test/browser-server";

test("only the previous label attaches to a browser server, at any verified directory", () => {
    // the output flow verifies a directory other than the app (its `served` argument), so the label
    // difference has to survive that too — and the verified directory must reach both labels alike.
    for (const dir of [undefined, ".", "/tmp/output-standalone"]) {
        const args = (label: "previous" | "candidate") =>
            dir === undefined
                ? verifyArgs(label, "ws://127.0.0.1:4242/abc")
                : verifyArgs(label, "ws://127.0.0.1:4242/abc", dir);
        const previous = args("previous");
        const candidate = args("candidate");
        expect(candidate).toContain(dir ?? ".");
        expect(previous).toContain(dir ?? ".");
        expect(previous).toContain("--connect");
        expect(previous[previous.indexOf("--connect") + 1]).toBe("ws://127.0.0.1:4242/abc");
        expect(candidate).not.toContain("--connect");
        // the verify arguments themselves are the same run on both sides; only the transport differs.
        expect(previous.slice(0, candidate.length)).toEqual(candidate);
    }
});

test("the 17 caller constructions keep explicit headed membership and ordinary headless membership", () => {
    const callers: Array<{ file: string; constructions: number; headed: boolean }> = [
        { file: "bench.ts", constructions: 3, headed: true },
        { file: "stall-attribution.ts", constructions: 1, headed: true },
        { file: "loaf-attribution.ts", constructions: 1, headed: true },
        { file: "flows.ts", constructions: 4, headed: false },
        { file: "recipes.ts", constructions: 1, headed: false },
        { file: "demos.ts", constructions: 1, headed: false },
        { file: "compile-concurrency.ts", constructions: 1, headed: false },
        { file: "physics-bench.ts", constructions: 1, headed: false },
        { file: "bench-tumble.ts", constructions: 1, headed: false },
        { file: "tumble-interaction.ts", constructions: 1, headed: false },
        { file: "boot-cost.ts", constructions: 1, headed: false },
    ];
    let total = 0;
    let explicit = 0;
    for (const caller of callers) {
        const source = readFileSync(resolve(import.meta.dir, caller.file), "utf8");
        const headedFlags = (source.match(/"--headed"/g) ?? []).length;
        expect(headedFlags).toBe(caller.headed ? caller.constructions : 0);
        if (!caller.headed) expect(source).not.toContain("skipReason");
        const calls = (source.match(/await verify(?:Batch)?\(/g) ?? []).length;
        const spawned = (source.match(/await spawnVerify\(/g) ?? []).length;
        expect(calls).toBe(caller.file === "boot-cost.ts" ? 0 : caller.constructions);
        expect(spawned).toBe(caller.file === "boot-cost.ts" ? 2 : 0);
        total += caller.constructions;
        explicit += headedFlags;
    }
    expect(total).toBe(16);
    expect(explicit).toBe(5);

    const ocean = JSON.parse(
        readFileSync(resolve(import.meta.dir, "../examples/showcase/ocean/package.json"), "utf8"),
    ) as { scripts?: { gate?: string } };
    expect(ocean.scripts?.gate).toBe("bunx shallot verify . --screenshot ocean.png");
    total += 1;
    expect(total).toBe(17);
});

test("the missing-crate accumulator requires failure, its diagnostic, and no raw ENOENT", () => {
    expect(
        missingCrateDiagnosticPass({
            ok: false,
            out: "corrupt install: reinstall @dylanebert/shallot",
        }),
    ).toBe(true);
    expect(
        missingCrateDiagnosticPass({
            ok: false,
            out: "cargo failed without the named remedy",
        }),
    ).toBe(false);
    expect(
        missingCrateDiagnosticPass({
            ok: true,
            out: "success, but corrupt install appeared in a warning",
        }),
    ).toBe(false);
    expect(
        missingCrateDiagnosticPass({
            ok: false,
            out: "corrupt install: ENOENT from cargo",
        }),
    ).toBe(false);
});

test("the endpoint reader refuses output with no ws:// line, naming the log", () => {
    expect(readEndpoint("ws://127.0.0.1:9/x\n", "/tmp/s.log")).toBe("ws://127.0.0.1:9/x");
    expect(readEndpoint("noise\n  ws://127.0.0.1:9/x  \nmore\n", "/tmp/s.log")).toBe(
        "ws://127.0.0.1:9/x",
    );
    // an empty or failed server must not hand "" to --connect and let verify red as though the
    // adapter were at fault.
    for (const output of ["", "Error: browser server crashed\n", "listening\n"]) {
        expect(() => readEndpoint(output, "/tmp/server.log")).toThrow("/tmp/server.log");
    }
});
