import { $ } from "bun";
import { resolve } from "node:path";
import { existsSync, mkdirSync } from "node:fs";

const root = resolve(import.meta.dir, "../rust");

// transforms (wasm-pack)
const transforms = resolve(root, "transforms");
await $`wasm-pack build --target web --release`.cwd(transforms);
// wasm-pack generates pkg/.gitignore with `*` which blocks npm/bun pack
// from including WASM files. Root .gitignore already handles git exclusion.
await $`rm -f ${resolve(transforms, "pkg/.gitignore")}`;
const transformsPkg = resolve(transforms, "pkg/package.json");
const pkg = await Bun.file(transformsPkg).json();
pkg.sideEffects = false;
await Bun.write(transformsPkg, JSON.stringify(pkg, null, 2) + "\n");
await $`bunx biome check --write ${resolve(transforms, "pkg")}/`.quiet();

// audio (cargo → wasm-opt)
const audio = resolve(root, "audio");
const audioPkg = resolve(audio, "pkg");
if (!existsSync(audioPkg)) mkdirSync(audioPkg);
await $`cargo build --target wasm32-unknown-unknown --release`.cwd(audio);
const audioWasm = resolve(audio, "target/wasm32-unknown-unknown/release/shallot_audio.wasm");
try {
    await $`wasm-opt -O3 --enable-nontrapping-float-to-int --enable-bulk-memory ${audioWasm} -o ${audioPkg}/shallot_audio.wasm`;
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
