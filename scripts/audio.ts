import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";
import { audioInputHash } from "../crates/audio/scripts/audio-inputs";
import { rustSourceRemaps } from "./rust-remap";
import { optimize } from "./wasm-opt";

const pkgRoot = resolve(import.meta.dir, "..");
const root = resolve(pkgRoot, "crates");

// audio (cargo → wasm-opt). A workspace member, so cargo writes to the root target/.
const audio = resolve(root, "audio");
const audioPkg = resolve(audio, "pkg");
if (!existsSync(audioPkg)) mkdirSync(audioPkg);
// RUSTFLAGS replaces the crate config, so repeat its SIMD and workspace-prefix flags.
const rustflags = [
    "-C target-feature=+simd128",
    "--remap-path-prefix=crates/audio/=",
    ...rustSourceRemaps(),
].join(" ");
await $`cargo build --target wasm32-unknown-unknown --release`
    .cwd(audio)
    .env({ ...process.env, RUSTFLAGS: rustflags });
const audioWasm = resolve(pkgRoot, "target/wasm32-unknown-unknown/release/shallot_audio.wasm");
await optimize(audioWasm, resolve(audioPkg, "shallot_audio.wasm"));
const inputHash = audioInputHash(pkgRoot);
await Bun.write(
    resolve(audioPkg, "shallot_audio.js"),
    `// Build inputs: sha256:${inputHash}
const url = new URL("shallot_audio.wasm", import.meta.url);
export default async function loadAudioWasm() {
    const response = await fetch(url);
    return response.arrayBuffer();
}
`,
);
await Bun.write(
    resolve(audioPkg, "shallot_audio.d.ts"),
    `export default function loadAudioWasm(): Promise<ArrayBuffer>;\n`,
);
