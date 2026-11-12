import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const DRACO_ARTIFACT = "src/extras/gltf/draco.wasm.ts";

export function dracoInputHash(root: string): string {
    return buildInputHash(root, [
        "Cargo.toml", "rust-toolchain.toml", "scripts/wasm-opt.ts", "scripts/rust-remap.ts", "scripts/wasm-memory.ts",
        "crates/build-inputs.ts", "crates/draco/Cargo.toml",
        "crates/draco/scripts/inputs.ts", "crates/draco/scripts/build.ts",
    ], ["crates/draco/src"], "shallot-draco");
}

export function dracoArtifactIsFresh(root: string): boolean {
    return buildArtifactIsFresh(root, DRACO_ARTIFACT, DRACO_ARTIFACT, dracoInputHash(root));
}
