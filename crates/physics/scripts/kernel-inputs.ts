import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const KERNEL_ARTIFACTS = [
    "src/standard/physics/kernel/kernel.wasm.ts",
    "src/standard/physics/kernel/kernel.shared.wasm.ts",
];
export const KERNEL_BUILD_COMMAND = "bun run crates/physics/scripts/build-kernel.ts";

export function kernelInputHash(root: string): string {
    return buildInputHash(
        root,
        [
            "Cargo.toml",
            "rust-toolchain.toml",
            "scripts/wasm-opt.ts",
            "scripts/rust-remap.ts",
            "crates/build-inputs.ts",
            "crates/physics/Cargo.toml",
            "crates/physics/.cargo/config.toml",
            "crates/physics/scripts/build-kernel.ts",
            "crates/physics/scripts/kernel-inputs.ts",
        ],
        ["crates/physics/src"],
        "shallot-physics",
    );
}

export function kernelArtifactsAreFresh(root: string): boolean {
    const inputHash = kernelInputHash(root);
    return KERNEL_ARTIFACTS.every((artifact) =>
        buildArtifactIsFresh(root, artifact, artifact, inputHash),
    );
}
