import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";

const root = resolve(import.meta.dir, "../rust");

// audio (cargo → wasm-opt)
const audio = resolve(root, "audio");
const audioPkg = resolve(audio, "pkg");
if (!existsSync(audioPkg)) mkdirSync(audioPkg);
await $`cargo build --target wasm32-unknown-unknown --release`.cwd(audio);
const audioWasm = resolve(audio, "target/wasm32-unknown-unknown/release/shallot_audio.wasm");
try {
    await $`wasm-opt -O3 --enable-simd --enable-nontrapping-float-to-int --enable-bulk-memory ${audioWasm} -o ${audioPkg}/shallot_audio.wasm`;
} catch {
    await $`cp ${audioWasm} ${audioPkg}/shallot_audio.wasm`;
}
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

// window (native host binary)
await $`cargo build --release`.cwd(resolve(root, "window"));

// docs
console.log("Building docs...");
await import("./docs");

// field docs (the editor's annotation-sourced UI reference data)
console.log("Generating field docs...");
await import("./fielddocs");

// starter example (regenerated from the create-shallot template)
console.log("Generating starter...");
await import("./starter");
