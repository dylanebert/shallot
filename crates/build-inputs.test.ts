import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { buildInputHash, cargoLockClosure } from "./build-inputs";

const lock = readFileSync(resolve(import.meta.dir, "../Cargo.lock"), "utf8");
// abf54055: adding cef-dll-sys to shallot-native staled both WASM stamps with identical bytes.
const unrelated = lock.replace(
    /(name = "shallot-native"\nversion = "[^"]+"\ndependencies = \[\n)/,
    '$1 "fixture-unrelated",\n',
);
const bumped = lock.replace(/(name = "serde_json"\nversion = ")([^"]+)"/, '$1$2-fixture"');

function hash(cargoLock: string, lockRoot: string): string {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-build-inputs-"));
    try {
        writeFileSync(resolve(root, "Cargo.lock"), cargoLock);
        writeFileSync(resolve(root, "bun.lock"), '    "binaryen": ["binaryen@1.0.0"],\n');
        return buildInputHash(root, [], [], lockRoot);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

test("a Cargo.lock edit outside a WASM crate's dependency closure leaves its build inputs unchanged", () => {
    expect(unrelated).not.toBe(lock);
    for (const crate of ["shallot-physics", "shallot-audio"])
        expect(hash(unrelated, crate)).toBe(hash(lock, crate));
});

test("a version change inside the kernel's dependency closure changes its build inputs", () => {
    expect(cargoLockClosure(lock, "shallot-physics").join()).toContain('name = "serde_json"');
    expect(hash(bumped, "shallot-physics")).not.toBe(hash(lock, "shallot-physics"));
});

test("a dependency the lock does not resolve to one package is refused", () => {
    expect(() => cargoLockClosure(lock, "fixture-absent")).toThrow(
        "Cargo.lock locks 0 packages for fixture-absent",
    );
});
