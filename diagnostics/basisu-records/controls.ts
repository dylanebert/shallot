import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import inputs from "../../crates/basisu/tests/records/inputs.json";
import { removeTestDirectory } from "../../scripts/test-directory";

const root = resolve(import.meta.dir, "../..");
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

export function readInputs() {
    return inputs.files.map((file) => {
        const bytes = readFileSync(resolve(root, "test-results/basisu-assets", file.path));
        if (digest(bytes) !== file.sha256) throw new Error(`${file.path}: digest mismatch`);
        return { path: file.path, bytes };
    });
}

/** Extracts verified no-supercompression controls outside the timed scope. */
export function controls() {
    const scratch = mkdtempSync(join(tmpdir(), "basisu-attribution-"));
    const records = new Map(readFileSync(resolve(root, "crates/basisu/tests/records/levels.tsv"), "utf8").trim().split("\n").slice(1).map((line) => {
        const [name, level, length, hash] = line.split("\t");
        return [`${name}\t${level}`, [Number(length), hash] as const];
    }));
    function plainKtx(bytes: Buffer<ArrayBuffer>, path: string): Buffer<ArrayBuffer> {
        if (bytes.readUInt32LE(44) !== 2) return bytes;
        const count = bytes.readUInt32LE(40);
        const end = Math.max(80 + count*24, bytes.readUInt32LE(48)+bytes.readUInt32LE(52), bytes.readUInt32LE(56)+bytes.readUInt32LE(60));
        const prefix = Buffer.from(bytes.subarray(0, end));
        prefix.writeUInt32LE(0, 44);
        let offset = Math.ceil(end/16)*16;
        const parts: Buffer<ArrayBuffer>[] = [prefix, Buffer.alloc(offset-end)];
        for (let level = 0; level < count; level++) {
            const output = join(scratch, "level.bin");
            const result = spawnSync(resolve(root, "test-results/basisu-reference/reference"), [path, String(level), output], { encoding: "utf8" });
            if (result.status !== 0) throw new Error("run generate.ts first to build the pinned native level extractor: " + result.stderr);
            const data = readFileSync(output);
            const name = path.split("/").at(-3) + "/" + path.split("/").at(-1)!.slice(0,-5);
            const expected = records.get(`${name}\t${level}`)!;
            if (data.length !== expected[0] || digest(data) !== expected[1]) throw new Error(`${name} level ${level}: extracted block digest mismatch`);
            prefix.writeBigUInt64LE(BigInt(offset), 80+level*24);
            prefix.writeBigUInt64LE(BigInt(data.length), 88+level*24);
            prefix.writeBigUInt64LE(BigInt(data.length), 96+level*24);
            const padding = (16-data.length%16)%16;
            parts.push(data, Buffer.alloc(padding));
            offset += data.length+padding;
        }
        return Buffer.concat(parts);
    }
    try {
        return readInputs().map((file) => ({
            ...file,
            plain: plainKtx(file.bytes, resolve(root, "test-results/basisu-assets", file.path)),
        }));
    } finally { removeTestDirectory(scratch); }
}
