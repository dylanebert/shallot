import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const GLTF_ARTIFACT = "src/extras/gltf/gltf.wasm.ts";

export function gltfInputHash(root: string): string {
    return buildInputHash(root, [
        "Cargo.toml", "rust-toolchain.toml", "scripts/wasm-opt.ts", "scripts/rust-remap.ts",
        "crates/build-inputs.ts", "crates/gltf/Cargo.toml",
        "crates/gltf/scripts/inputs.ts", "crates/gltf/scripts/build.ts",
    ], ["crates/gltf/src"], "shallot-gltf");
}

export function gltfArtifactIsFresh(root: string): boolean {
    return buildArtifactIsFresh(root, GLTF_ARTIFACT, GLTF_ARTIFACT, gltfInputHash(root));
}
