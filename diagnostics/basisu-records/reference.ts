import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import inputs from "../../crates/basisu/tests/records/inputs.json";

export interface ReferenceImage {
    getLevels(): number;
    getHasAlpha(): boolean;
    startTranscoding(): boolean;
    getImageTranscodedSizeInBytes(level: number, layer: number, face: number, format: number): number;
    transcodeImage(dst: Uint8Array, level: number, layer: number, face: number, format: number, alpha: number, channel0: number, channel1: number): boolean;
    close(): void;
    delete(): void;
}
export function referenceDirectory(): string {
    const path = process.env.BASIS_UNIVERSAL;
    if (!path) throw new Error("set BASIS_UNIVERSAL to the pinned Basis Universal checkout");
    function git(...args: string[]) {
        const result = spawnSync("git", ["-C", path!, ...args], { encoding: "utf8" });
        if (result.status !== 0) throw new Error(result.stderr);
        return result.stdout.trim();
    }
    if (git("rev-parse", "HEAD") !== inputs.decoder) throw new Error("Basis Universal pin mismatch");
    if (git("diff", "3784d52", inputs.decoder, "--", "transcoder", "zstd", "webgl/build.sh", "webgl/transcoder/basis_wrappers.cpp", "webgl/transcoder/CMakeLists.txt")) throw new Error("committed reference build's transcoder inputs differ from the pin");
    if (git("status", "--porcelain", "--untracked-files=no")) throw new Error("reference checkout is dirty");
    return path;
}
export async function loadReference() {
    const path = referenceDirectory();
    const factory = createRequire(import.meta.path)(resolve(path, "webgl/transcoder/build/basis_transcoder.js"));
    const module: { initializeBasis(): void; KTX2File: new (bytes: Uint8Array) => ReferenceImage } = await factory({ wasmBinary: readFileSync(resolve(path, "webgl/transcoder/build/basis_transcoder.wasm")) });
    module.initializeBasis();
    return module;
}
