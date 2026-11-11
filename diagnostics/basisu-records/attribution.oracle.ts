import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import inputs from "../../crates/basisu/tests/records/inputs.json";
import { bunChild } from "../../scripts/bun-child";
import { removeTestDirectory } from "../../scripts/test-directory";
import { openBasisImage, type BasisTarget } from "../../src/extras/gltf/basisu";
import { loadReference } from "./reference";

const root = resolve(import.meta.dir, "../..");
const sync = Bun.spawnSync({ ...bunChild([resolve(import.meta.dir, "records.ts")]), cwd: root, stdout: "pipe", stderr: "pipe" });
if (sync.exitCode !== 0) throw new Error(new TextDecoder().decode(sync.stderr));
const reference = await loadReference();
const scratch = mkdtempSync(join(tmpdir(), "basisu-attribution-"));
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const records = new Map(readFileSync(resolve(root, "crates/basisu/tests/records/levels.tsv"), "utf8").trim().split("\n").slice(1).map((line) => {
    const [name, level, length, hash] = line.split("\t");
    return [`${name}\t${level}`, [Number(length), hash] as const];
}));

function plainKtx(bytes: Buffer, path: string): Buffer {
    if (bytes.readUInt32LE(44) !== 2) return bytes;
    const count = bytes.readUInt32LE(40);
    const end = Math.max(80 + count*24, bytes.readUInt32LE(48)+bytes.readUInt32LE(52), bytes.readUInt32LE(56)+bytes.readUInt32LE(60));
    const prefix = Buffer.from(bytes.subarray(0, end));
    prefix.writeUInt32LE(0, 44);
    let offset = Math.ceil(end/16)*16;
    const parts: Buffer[] = [prefix, Buffer.alloc(offset-end)];
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
interface Split { setup: number; teardown: number; copies: number; etc1: number; "etc2-rgba": number; astc: number }
const zero = (): Split => ({ setup: 0, teardown: 0, copies: 0, etc1: 0, "etc2-rgba": 0, astc: 0 });
const median = (values: number[]) => { values.sort((a,b) => a-b); return (values[3]+values[4])/2; };
try {
    for (const model of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
        const files = inputs.files.filter((f) => f.path.startsWith(`${model}/`)).map((file) => {
            const path = resolve(root, "test-results/basisu-assets", file.path);
            const bytes = readFileSync(path);
            if (digest(bytes) !== file.sha256) throw new Error(`${file.path}: digest mismatch`);
            const own = openBasisImage(bytes);
            const ref = new reference.KTX2File(bytes);
            const target: BasisTarget = own.model === "uastc" ? "astc" : ref.getHasAlpha() ? "etc2-rgba" : "etc1";
            const levels = own.levels;
            own.dispose(); ref.close(); ref.delete();
            return { bytes, plain: plainKtx(bytes,path), target: target as "astc"|"etc1"|"etc2-rgba", levels, format: target === "astc" ? 10 : target === "etc2-rgba" ? 1 : 0 };
        });
        // Validate the no-supercompression control through both JavaScript entries.
        for (const f of files) {
            const original = openBasisImage(f.bytes);
            const plain = openBasisImage(f.plain);
            const ref = new reference.KTX2File(f.plain);
            if (!ref.startTranscoding()) throw new Error("reference start failed");
            try {
                for (let level = 0; level < f.levels; level++) {
                    const output = new Uint8Array(ref.getImageTranscodedSizeInBytes(level,0,0,f.format));
                    if (!ref.transcodeImage(output,level,0,0,f.format,0,-1,-1)) throw new Error("reference transcode failed");
                    const hash = digest(output);
                    if (digest(original.transcode(level,f.target)) !== hash || digest(plain.transcode(level,f.target)) !== hash) throw new Error("plain KTX2 control changed output");
                }
            } finally { original.dispose(); plain.dispose(); ref.close(); ref.delete(); }
        }
        let copyTime = 0;
        const set = Uint8Array.prototype.set;
        const slice = Uint8Array.prototype.slice;
        function run(own: boolean, plain: boolean): Split {
            const split = zero();
            copyTime = 0;
            Uint8Array.prototype.set = function (source, offset) {
                const start = performance.now();
                set.call(this,source,offset);
                copyTime += performance.now()-start;
            };
            Uint8Array.prototype.slice = function (start, end) {
                const before = performance.now();
                const value = slice.call(this,start,end);
                copyTime += performance.now()-before;
                return value;
            };
            try {
                for (const f of files) {
                    const bytes = plain ? f.plain : f.bytes;
                    let start = performance.now();
                    let beforeCopies = copyTime;
                    if (own) {
                        const image = openBasisImage(bytes);
                        split.setup += performance.now()-start-(copyTime-beforeCopies);
                        try {
                            start = performance.now(); beforeCopies = copyTime;
                            for (let level = 0; level < f.levels; level++) image.transcode(level,f.target);
                            split[f.target] += performance.now()-start-(copyTime-beforeCopies);
                        } finally { const start=performance.now(); image.dispose(); split.teardown+=performance.now()-start; }
                    } else {
                        const image = new reference.KTX2File(bytes);
                        if (!image.startTranscoding()) throw new Error("reference start failed");
                        split.setup += performance.now()-start-(copyTime-beforeCopies);
                        try {
                            start = performance.now(); beforeCopies = copyTime;
                            for (let level = 0; level < f.levels; level++) {
                                const output = new Uint8Array(image.getImageTranscodedSizeInBytes(level,0,0,f.format));
                                if (!image.transcodeImage(output,level,0,0,f.format,0,-1,-1)) throw new Error("reference transcode failed");
                            }
                            split[f.target] += performance.now()-start-(copyTime-beforeCopies);
                        } finally { const start=performance.now(); image.close(); image.delete(); split.teardown+=performance.now()-start; }
                    }
                }
                split.copies = copyTime;
                return split;
            } finally { Uint8Array.prototype.set = set; Uint8Array.prototype.slice = slice; }
        }
        const samples: Split[][] = [[],[],[],[]];
        for (let round = 0; round < 12; round++) {
            for (let i = 0; i < 4; i++) {
                const index = (round+i)%4;
                const result = run(index<2,(index&1)!==0);
                if (round>=4) samples[index].push(result);
            }
        }
        const summary = samples.map((samples) => Object.fromEntries(Object.keys(zero()).map((key) => [key,median(samples.map((s) => s[key as keyof Split]))])) as unknown as Split);
        for (const [name,index] of [["own",0],["reference",2]] as const) {
            const original = summary[index]; const plain = summary[index+1];
            console.log(JSON.stringify({ model, implementation: name, setupMs: original.setup, teardownMs: original.teardown, boundaryCopiesMs: original.copies, etc1Ms: original.etc1, etc2RgbaMs: original["etc2-rgba"], astcWithoutZstdMs: plain.astc, zstdDeltaMs: original.astc-plain.astc }));
        }
    }
} finally { removeTestDirectory(scratch); }
