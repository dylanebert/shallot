import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";
import { optimize } from "./wasm-opt";

const root = resolve(import.meta.dir, "../rust");

// audio (cargo → wasm-opt)
const audio = resolve(root, "audio");
const audioPkg = resolve(audio, "pkg");
if (!existsSync(audioPkg)) mkdirSync(audioPkg);
await $`cargo build --target wasm32-unknown-unknown --release`.cwd(audio);
const audioWasm = resolve(audio, "target/wasm32-unknown-unknown/release/shallot_audio.wasm");
await optimize(audioWasm, resolve(audioPkg, "shallot_audio.wasm"));
await Bun.write(
    resolve(audioPkg, "shallot_audio.js"),
    `const url = new URL("shallot_audio.wasm", import.meta.url);
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
