import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const BASISU_ARTIFACT = "src/extras/gltf/basisu.wasm.ts";

export function basisuInputHash(root: string): string {
    return buildInputHash(root, [
        "Cargo.toml", "rust-toolchain.toml", "scripts/wasm-opt.ts", "scripts/rust-remap.ts", "scripts/wasm-memory.ts",
        "crates/build-inputs.ts", "crates/basisu/Cargo.toml",
        "crates/basisu/scripts/inputs.ts", "crates/basisu/scripts/build.ts",
    ], ["crates/basisu/src"], "shallot-basisu");
}

export function basisuArtifactIsFresh(root: string): boolean {
    return buildArtifactIsFresh(root, BASISU_ARTIFACT, BASISU_ARTIFACT, basisuInputHash(root));
}
