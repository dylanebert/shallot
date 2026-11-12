import { expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { pathToFileURL } from "node:url";
import inputs from "../../crates/gltf/tests/meshopt/inputs.json";
import { readInput } from "./records";

test("missing corpus inputs name the pinned file", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "meshopt-input-"));
    try {
        await expect(readInput(inputs.files[0], join(scratch, "cache"), pathToFileURL(`${scratch}/`))).rejects.toThrow(inputs.files[0].path);
    } finally { rmSync(scratch, { recursive: true, force: true }); }
});

test("a mismatched corpus input is refused by path and SHA-256", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "meshopt-input-"));
    const file = inputs.files[0];
    try {
        const path = join(scratch, file.path);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, "corrupt");
        await expect(readInput(file, scratch)).rejects.toThrow(`${file.path}: SHA-256 mismatch`);
    } finally { rmSync(scratch, { recursive: true, force: true }); }
});
