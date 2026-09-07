// Arms for the frozen previous-release baseline. Each one is a way a later boundary move can hollow the
// fixture out while every other gate stays green — a deleted scaffold, a truncated tarball inventory, a
// digest edited to match whatever was fetched.

import { afterEach, expect, test } from "bun:test";
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { checkPin, FIXTURE_DIR, subresourceIntegrity } from "./check-compat-pin";

const REPO = resolve(import.meta.dir, "..");
const trees: string[] = [];
afterEach(() => {
    for (const tree of trees.splice(0)) rmSync(tree, { recursive: true, force: true });
});

/** a copy of the real fixture, so a mutation is applied to the thing the gate actually reads. */
const copy = (): string => {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-compat-pin-"));
    trees.push(root);
    mkdirSync(dirname(resolve(root, FIXTURE_DIR)), { recursive: true });
    cpSync(resolve(REPO, FIXTURE_DIR), resolve(root, FIXTURE_DIR), { recursive: true });
    return root;
};

const rewritePin = (root: string, edit: (pin: Record<string, never>) => void): void => {
    const path = resolve(root, FIXTURE_DIR, "PIN.json");
    const pin = JSON.parse(require("node:fs").readFileSync(path, "utf8"));
    edit(pin);
    writeFileSync(path, JSON.stringify(pin, null, 4));
};

test("the committed baseline is consistent", () => {
    expect(checkPin(REPO)).toEqual([]);
});

test("a missing pin refuses instead of reading as an empty baseline", () => {
    const root = copy();
    rmSync(resolve(root, FIXTURE_DIR, "PIN.json"));
    expect(checkPin(root).join("\n")).toContain("PIN.json is missing");
});

test("a deleted scaffold refuses", () => {
    const root = copy();
    rmSync(resolve(root, FIXTURE_DIR, "scaffold"), { recursive: true });
    expect(checkPin(root).join("\n")).toContain("holds no emitted files");
});

test("a scaffold that lost its emitted manifest refuses", () => {
    const root = copy();
    rmSync(resolve(root, FIXTURE_DIR, "scaffold/shallot.json"));
    expect(checkPin(root).join("\n")).toContain("missing the emitted shallot.json");
});

test("an inventory that no longer matches the recorded file count refuses", () => {
    const root = copy();
    writeFileSync(resolve(root, FIXTURE_DIR, "engine-files.txt"), "package.json\n");
    expect(checkPin(root).join("\n")).toContain("lists 1 file(s); the pin records 424");
});

test("a dropped baseline package refuses", () => {
    const root = copy();
    rewritePin(root, (pin) => {
        delete (pin as Record<string, Record<string, unknown>>).packages["create-shallot"];
    });
    expect(checkPin(root).join("\n")).toContain("pin names no create-shallot baseline");
});

test("a digest that is not a sha512 subresource refuses", () => {
    const root = copy();
    rewritePin(root, (pin) => {
        (pin as Record<string, Record<string, Record<string, string>>>).packages[
            "@dylanebert/shallot"
        ].integrity = "sha256-nope";
    });
    expect(checkPin(root).join("\n")).toContain("not a sha512 subresource digest");
});

test("a tarball url that names a different version refuses", () => {
    const root = copy();
    rewritePin(root, (pin) => {
        (pin as Record<string, Record<string, Record<string, string>>>).packages[
            "@dylanebert/shallot"
        ].version = "0.9.4";
    });
    expect(checkPin(root).join("\n")).toContain("does not name version 0.9.4");
});

// the digest helper the --fetch arm compares with, against a value derived independently of it
test("subresourceIntegrity spells a sha512 digest the way npm records it", () => {
    const bytes = new TextEncoder().encode("shallot");
    const expected = `sha512-${require("node:crypto").createHash("sha512").update("shallot").digest("base64")}`;
    expect(subresourceIntegrity(bytes)).toBe(expected);
    expect(subresourceIntegrity(bytes)).toMatch(/^sha512-[A-Za-z0-9+/]+=*$/);
});
