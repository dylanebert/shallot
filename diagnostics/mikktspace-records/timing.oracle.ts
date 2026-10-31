// MIKKTSPACE=/path/to/pinned/MikkTSpace bun diagnostics/mikktspace-records/timing.oracle.ts
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import records from "../../src/core/mesh/fixtures/mikktspace.json";

function run(args: string[]) {
    const result = Bun.spawnSync(args);
    if (result.exitCode !== 0) throw new Error(`${args.join(" ")}: ${result.stderr.toString()}`);
    return result.stdout.toString().trim();
}
const checkout = process.env.MIKKTSPACE;
if (!checkout) throw new Error("Set MIKKTSPACE to the pinned C checkout");
if (run(["git", "-C", checkout, "rev-parse", "HEAD"]) !== records.revision) throw new Error("MikkTSpace revision differs from records");
run(["git", "-C", checkout, "diff", "--exit-code", "HEAD", "--", "mikktspace.c", "mikktspace.h"]);
const temporary = mkdtempSync(resolve(tmpdir(), "shallot-mikktspace-timing-"));
try {
    // An independent Cargo root: bevy_mikktspace never enters Shallot's workspace or lockfile.
    writeFileSync(resolve(temporary, "Cargo.toml"), `[package]\nname="mikktspace-timing"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\nbevy_mikktspace="=1.0.0"\n[[bin]]\nname="timing"\npath="main.rs"\n[profile.release]\nopt-level=3\n`);
    copyFileSync(resolve(import.meta.dir, "timing.rs"), resolve(temporary, "main.rs"));
    copyFileSync(resolve(import.meta.dir, "../../crates/mikktspace/src/tangent.rs"), resolve(temporary, "tangent.rs"));
    run(["cargo", "build", "--release", "--manifest-path", resolve(temporary, "Cargo.toml")]);
    const rust = resolve(temporary, "target/release/timing");
    const c = resolve(temporary, "reference");
    run([process.env.CC ?? "clang", "-O3", "-ffp-contract=off", "-I", checkout, resolve(import.meta.dir, "reference.c"), resolve(checkout, "mikktspace.c"), "-lm", "-o", c]);
    console.log(run(["sysctl", "-n", "machdep.cpu.brand_string"]));
    console.log(run(["rustc", "--version"]));
    console.log(run([process.env.CC ?? "clang", "--version"]).split("\n")[0]);
    console.log("Native release kernels, same triangle meshes; I/O, decode and remapping excluded. Five warmups; seven alternating samples; medians in microseconds. Rust includes output allocation/free; C reuses its output buffer. No timing assertion.");
    for (const record of records.records) {
        const path = resolve(import.meta.dir, "../../src/core/mesh/fixtures", record.file.replace(".bin", ".input.bin"));
        const bytes = readFileSync(path);
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const [nv, nc] = words;
        const vertices = new Float32Array(words.buffer, words.byteOffset + 8, nv * 8);
        const positions = new Float32Array(nv * 3), normals = new Float32Array(nv * 3), uv = new Float32Array(nv * 2);
        for (let i = 0; i < nv; i++) {
            positions.set(vertices.subarray(i * 8, i * 8 + 3), i * 3);
            normals.set(vertices.subarray(i * 8 + 4, i * 8 + 7), i * 3);
            uv.set([vertices[i * 8 + 3], vertices[i * 8 + 7]], i * 2);
        }
        const input = resolve(temporary, "input"), output = resolve(temporary, "output");
        writeFileSync(input, Buffer.concat([bytes.subarray(0, 8), ...[positions, normals, uv].map(a => Buffer.from(a.buffer)), bytes.subarray(8 + nv * 32)]));
        const iterations = String(Math.max(20, Math.floor(200000 / nc)));
        const timings: Record<string, number[]> = { port: [], bevy: [], c: [] };
        for (let sample = 0; sample < 7; sample++) {
            for (const engine of sample % 2 ? ["c", "bevy", "port"] : ["port", "bevy", "c"]) {
                const result = engine === "c" ? run([c, input, output, iterations]) : run([rust, path, engine, iterations]);
                timings[engine].push(Number(result));
            }
        }
        const median = (xs: number[]) => xs.sort((a,b) => a-b)[3];
        console.log(JSON.stringify({mesh: record.name, corners: nc, iterations: Number(iterations), port: median(timings.port), bevy: median(timings.bevy), c: median(timings.c)}));
    }
} finally {
    rmSync(temporary, { recursive: true, force: true });
}
