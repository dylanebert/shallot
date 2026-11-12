import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import inputs from "../../crates/draco/tests/records/inputs.json";
import type { Reference, Stream } from "./comparison";

const root = resolve(import.meta.dir, "../..");
/** The pinned clean Draco checkout, whose javascript/ holds Google's committed glTF decoder build. */
export function referenceDirectory(): string {
    const path = process.env.DRACO;
    if (!path) throw new Error("set DRACO to the pinned Draco checkout");
    const git = (...args: string[]) => {
        const result = spawnSync("git", ["-C", path, ...args], { encoding: "utf8" });
        if (result.status !== 0) throw new Error(result.stderr);
        return result.stdout.trim();
    };
    if (git("rev-parse", "HEAD") !== inputs.decoder) throw new Error("Draco pin mismatch");
    if (git("status", "--porcelain", "--untracked-files=no")) throw new Error("reference checkout is dirty");
    return path;
}
export const WRAPPER = "javascript/draco_wasm_wrapper_gltf.js";
export const WASM = "javascript/draco_decoder_gltf.wasm";
export async function loadReference(): Promise<Reference> {
    const path = referenceDirectory();
    const factory = createRequire(import.meta.path)(resolve(path, WRAPPER));
    return await factory({ wasmBinary: readFileSync(resolve(path, WASM)) });
}
/** Every corpus stream, hash-verified, with its recorded counts and its accessors' requests. */
export function readStreams(): Stream[] {
    const sync = Bun.spawnSync({ cmd: [process.execPath, resolve(import.meta.dir, "records.ts")], cwd: root, stdout: "pipe", stderr: "pipe" });
    if (sync.exitCode !== 0) throw new Error(sync.stderr.toString());
    const directory = sync.stdout.toString().trim();
    const rows = (name: string) => readFileSync(resolve(root, `crates/draco/tests/records/${name}.tsv`), "utf8").trim().split("\n").filter((l) => !l.startsWith("#")).map((l) => l.split("\t"));
    const attributes = rows("attributes");
    return rows("faces").filter(([name]) => !name.startsWith("fixtures/")).map(([name, faces, , , , sha256, , points]) => {
        const bytes = new Uint8Array(readFileSync(resolve(directory, `${name}.drc`)));
        if (createHash("sha256").update(bytes).digest("hex") !== sha256) throw new Error(`${name}: digest mismatch`);
        // Every corpus row requests its accessor's type, which is the decoded type.
        const requests = attributes.filter((row) => row[0] === name).map((row) => [Number(row[2]), Number(row[7])] as [number, number]);
        return { name, bytes, faces: Number(faces), points: Number(points), requests };
    });
}
