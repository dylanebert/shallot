import { expect, test } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { audioInputHash } from "./audio/scripts/audio-inputs";
import { cargoLockClosure } from "./build-inputs";
import { kernelInputHash } from "./physics/scripts/kernel-inputs";

const lock = (nativeDeps: string, serdeVersion = "1.0.0") => `version = 4

[[package]]
name = "serde_json"
version = "${serdeVersion}"

[[package]]
name = "shallot-native"
version = "0.10.0"
dependencies = [
${nativeDeps}
]

[[package]]
name = "shallot-physics"
version = "0.10.0"
dependencies = [
 "serde_json",
]
`;

const repo = resolve(import.meta.dir, "..");
const crates = [
    { name: "kernel", hash: kernelInputHash },
    { name: "audio", hash: audioInputHash },
];

/** The real tree's inputs, with its Cargo.lock rewritten. */
function hashWithLock(hash: (root: string) => string, edit: (lock: string) => string): string {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-build-inputs-"));
    try {
        for (const path of ["Cargo.toml", "rust-toolchain.toml", "bun.lock", "scripts", "crates"])
            cpSync(resolve(repo, path), resolve(root, path), {
                recursive: true,
                filter: (source) => !/[\\/](target|node_modules|pkg)$/.test(source),
            });
        writeFileSync(resolve(root, "Cargo.lock"), edit(readFileSync(resolve(repo, "Cargo.lock"), "utf8")));
        return hash(root);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}

for (const crate of crates) {
    test(`a Cargo.lock edit outside the ${crate.name} dependency closure leaves its build inputs unchanged`, () => {
        // abf54055: adding cef-dll-sys to shallot-native staled both WASM stamps with identical bytes.
        const unrelated = (lock: string) =>
            lock.replace(/(name = "shallot-native"\nversion = "[^"]+"\ndependencies = \[\n)/, '$1 "fixture-unrelated",\n');
        expect(hashWithLock(crate.hash, unrelated)).toBe(hashWithLock(crate.hash, (lock) => lock));
    });
}

test("a version change inside the kernel dependency closure changes its build inputs", () => {
    const bump = (lock: string) =>
        lock.replace(/(name = "serde_json"\nversion = ")([^"]+)"/, '$1$2-fixture"');
    expect(hashWithLock(kernelInputHash, bump)).not.toBe(hashWithLock(kernelInputHash, (lock) => lock));
});

test("a dependency the lock does not resolve to one package is refused", () => {
    expect(() => cargoLockClosure(lock(""), "shallot-audio")).toThrow(
        "Cargo.lock locks 0 packages for shallot-audio",
    );
});
