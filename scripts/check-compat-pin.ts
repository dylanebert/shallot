#!/usr/bin/env bun
/**
 * check-compat-pin — the frozen previous-release baseline is internally consistent, and (with
 * `--fetch`) the registry still serves the exact bytes it pins.
 *
 * The offline arm is what `bun run check` runs: it proves the pin names real frozen inputs and that the
 * recorded file count matches the recorded inventory, so a fixture half-deleted by a later move refuses
 * instead of quietly comparing against nothing. The `--fetch` arm downloads both tarballs and recomputes
 * their digests; it needs the network, so it is an operator command, not a gate.
 *
 * @example
 *   bun run scripts/check-compat-pin.ts            # offline consistency
 *   bun run scripts/check-compat-pin.ts --fetch    # also re-verify the registry bytes
 */

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

export interface PinnedPackage {
    version: string;
    tarball: string;
    integrity: string;
    shasum: string;
    fileCount: number;
    unpackedSize: number;
}

export interface CompatPin {
    why: string;
    capturedFrom: string;
    packages: Record<string, PinnedPackage>;
}

export const FIXTURE_DIR = "scripts/install-test/compat-0.9.5";

/** Every way the frozen baseline can be inconsistent with itself. Pure over the tree so the clauses are
 *  fixture-testable without a network. */
export function checkPin(root: string): string[] {
    const dir = resolve(root, FIXTURE_DIR);
    const errors: string[] = [];
    const pinPath = resolve(dir, "PIN.json");
    if (!existsSync(pinPath)) return [`${FIXTURE_DIR}/PIN.json is missing`];
    const pin = JSON.parse(readFileSync(pinPath, "utf8")) as CompatPin;

    const names = Object.keys(pin.packages);
    for (const expected of ["@dylanebert/shallot", "create-shallot"]) {
        if (!names.includes(expected)) errors.push(`pin names no ${expected} baseline`);
    }
    for (const [name, p] of Object.entries(pin.packages)) {
        if (!/^sha512-[A-Za-z0-9+/]+=*$/.test(p.integrity))
            errors.push(`${name}: integrity is not a sha512 subresource digest`);
        if (!/^[0-9a-f]{40}$/.test(p.shasum)) errors.push(`${name}: shasum is not a sha1 digest`);
        if (!p.tarball.endsWith(`-${p.version}.tgz`))
            errors.push(`${name}: tarball url does not name version ${p.version}`);
        if (!(p.fileCount > 0) || !(p.unpackedSize > 0))
            errors.push(`${name}: fileCount and unpackedSize must both be positive`);
    }

    // The inventory is the thing a later move actually breaks: an export or packed artifact that stops
    // shipping shows up as a diff only while this list is complete and matches the recorded count.
    const inventory = resolve(dir, "engine-files.txt");
    if (!existsSync(inventory)) errors.push(`${FIXTURE_DIR}/engine-files.txt is missing`);
    else {
        const lines = readFileSync(inventory, "utf8").split("\n").filter(Boolean);
        const declared = pin.packages["@dylanebert/shallot"]?.fileCount;
        if (declared !== undefined && lines.length !== declared)
            errors.push(
                `engine-files.txt lists ${lines.length} file(s); the pin records ${declared}`,
            );
    }
    if (!existsSync(resolve(dir, "engine-package.json")))
        errors.push(`${FIXTURE_DIR}/engine-package.json is missing`);

    // The scaffold is the original emitted project. An empty tree would let the compatibility arm pass
    // by comparing nothing.
    const scaffold = resolve(dir, "scaffold");
    const emitted = existsSync(scaffold)
        ? readdirSync(scaffold, { recursive: true, withFileTypes: true }).filter((e) => e.isFile())
        : [];
    if (emitted.length === 0) errors.push(`${FIXTURE_DIR}/scaffold holds no emitted files`);
    for (const required of ["package.json", "shallot.json"]) {
        if (!existsSync(resolve(scaffold, required)))
            errors.push(`${FIXTURE_DIR}/scaffold is missing the emitted ${required}`);
    }
    return errors;
}

/** sha512 subresource digest of `bytes`, in the `sha512-<base64>` spelling npm records. */
export function subresourceIntegrity(bytes: Uint8Array): string {
    return `sha512-${createHash("sha512").update(bytes).digest("base64")}`;
}

async function fetchAndVerify(pin: CompatPin): Promise<string[]> {
    const errors: string[] = [];
    for (const [name, p] of Object.entries(pin.packages)) {
        const response = await fetch(p.tarball);
        if (!response.ok) {
            errors.push(`${name}: ${p.tarball} returned ${response.status}`);
            continue;
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        const integrity = subresourceIntegrity(bytes);
        const shasum = createHash("sha1").update(bytes).digest("hex");
        if (integrity !== p.integrity) errors.push(`${name}: integrity mismatch (${integrity})`);
        if (shasum !== p.shasum) errors.push(`${name}: shasum mismatch (${shasum})`);
        if (errors.length === 0) console.log(`  ✓ ${name}@${p.version} bytes match the pin`);
    }
    return errors;
}

if (import.meta.main) {
    const root = resolve(import.meta.dir, "..");
    const errors = checkPin(root);
    if (errors.length === 0 && process.argv.includes("--fetch")) {
        const pin = JSON.parse(
            readFileSync(resolve(root, FIXTURE_DIR, "PIN.json"), "utf8"),
        ) as CompatPin;
        errors.push(...(await fetchAndVerify(pin)));
    }
    if (errors.length > 0) {
        console.error(errors.map((e) => `✗ ${e}`).join("\n"));
        process.exit(1);
    }
    console.log("✓ frozen 0.9.5 compatibility baseline consistent");
}
