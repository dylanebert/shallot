// MIKKTSPACE=/path/to/MikkTSpace bun diagnostics/mikktspace-records/generate.ts
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";
import { assets, models, primitives } from "./inputs";

const revision = "3e895b49d05ea07e4c2133156cfa94369e19e409";
const checkout = process.env.MIKKTSPACE;
if (!checkout) throw new Error(`set MIKKTSPACE to a checkout at ${revision}`);
function run(args: string[]): string {
    const result = Bun.spawnSync(args);
    if (result.exitCode !== 0) throw new Error(`${args.join(" ")}: ${result.stderr.toString()}`);
    return result.stdout.toString().trim();
}
const head = run(["git", "-C", checkout, "rev-parse", "HEAD"]);
if (head !== revision) throw new Error(`MIKKTSPACE is at ${head}, not ${revision}`);
run(["git", "-C", checkout, "diff", "--exit-code", "HEAD", "--", "mikktspace.c", "mikktspace.h"]);
await syncFiles(manifest.models.filter(model => models.includes(model.name)).flatMap(model => model.files), new URL(`${manifest.commit}/`, manifest.source), assets);
const temporary = mkdtempSync(resolve(tmpdir(), "shallot-mikktspace-records-"));
try {
    const executable = resolve(temporary, "reference");
    run([process.env.CC ?? "clang", "-O3", "-ffp-contract=off", "-I", checkout, resolve(import.meta.dir, "reference.c"), resolve(checkout, "mikktspace.c"), "-lm", "-o", executable]);
    const records = [];
    for (const name of models) for (const p of primitives(name)) {
        const count = p.vertices.length / 8;
        const positions = new Float32Array(count * 3), normals = new Float32Array(count * 3), uv = new Float32Array(count * 2);
        for (let i = 0; i < count; i++) {
            positions.set(p.vertices.subarray(i * 8, i * 8 + 3), i * 3);
            normals.set(p.vertices.subarray(i * 8 + 4, i * 8 + 7), i * 3);
            uv.set([p.vertices[i * 8 + 3], p.vertices[i * 8 + 7]], i * 2);
        }
        const input = resolve(temporary, "input"), output = resolve(temporary, "output");
        writeFileSync(input, Buffer.concat([Buffer.from(new Uint32Array([count, p.indices.length]).buffer), ...[positions, normals, uv, p.indices].map(array => Buffer.from(array.buffer))]));
        run([executable, input, output]);
        const file = `${name}-${p.mesh}-${p.primitive}.bin`;
        const bytes = readFileSync(output);
        writeFileSync(resolve(import.meta.dir, "../../src/core/mesh/fixtures", file.replace(".bin", ".input.bin")), Buffer.concat([Buffer.from(new Uint32Array([count, p.indices.length]).buffer), Buffer.from(p.vertices.buffer), Buffer.from(p.indices.buffer)]));
        writeFileSync(resolve(import.meta.dir, "../../src/core/mesh/fixtures", file), bytes);
        records.push({ name, mesh: p.mesh, primitive: p.primitive, corners: p.indices.length, file });
    }
    writeFileSync(resolve(import.meta.dir, "../../src/core/mesh/fixtures/mikktspace.json"), `${JSON.stringify({ revision, assetsCommit: manifest.commit, format: "Little-endian float32 xyzw per triangle corner; MikkTSpace w negated for glTF.", records }, null, 4)}\n`);
    console.log(`Recorded ${records.length} triangle primitives from MikkTSpace ${revision}`);
} finally {
    rmSync(temporary, { recursive: true, force: true });
}
