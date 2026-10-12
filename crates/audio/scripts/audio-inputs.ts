import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { buildArtifactIsFresh, buildInputHash } from "../../build-inputs";

export const AUDIO_ARTIFACT = "crates/audio/pkg/shallot_audio.wasm";
export const AUDIO_STAMP_FILE = "crates/audio/pkg/shallot_audio.js";
export const AUDIO_BUILD_COMMAND = "bun scripts/audio.ts";

export function audioInputHash(root: string): string {
    return buildInputHash(
        root,
        [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "scripts/wasm-opt.ts",
            "scripts/rust-remap.ts",
            "scripts/audio.ts",
            "crates/build-inputs.ts",
            "crates/audio/Cargo.toml",
            "crates/audio/.cargo/config.toml",
            "crates/audio/scripts/audio-inputs.ts",
        ],
        ["crates/audio/src"],
    );
}

export function audioArtifactIsFresh(root: string): boolean {
    return (
        existsSync(resolve(root, "crates/audio/pkg/shallot_audio.d.ts")) &&
        buildArtifactIsFresh(
            root,
            AUDIO_ARTIFACT,
            AUDIO_STAMP_FILE,
            audioInputHash(root),
            0,
        )
    );
}
