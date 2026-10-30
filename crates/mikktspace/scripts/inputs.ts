import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const MIKKTSPACE_ARTIFACT = "src/core/mesh/mikktspace.wasm.ts";

export function mikktspaceInputHash(root: string): string {
    return buildInputHash(root, [
        "Cargo.toml", "rust-toolchain.toml", "scripts/wasm-opt.ts", "scripts/rust-remap.ts",
        "crates/build-inputs.ts", "crates/mikktspace/Cargo.toml",
        "crates/mikktspace/scripts/inputs.ts", "crates/mikktspace/scripts/build.ts",
    ], ["crates/mikktspace/src"], "shallot-mikktspace");
}

export function mikktspaceArtifactIsFresh(root: string): boolean {
    return buildArtifactIsFresh(root, MIKKTSPACE_ARTIFACT, MIKKTSPACE_ARTIFACT, mikktspaceInputHash(root));
}
