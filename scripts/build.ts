import { resolve } from "node:path";
import { $ } from "bun";
import { audioArtifactIsFresh } from "../crates/audio/scripts/audio-inputs";
import { kernelArtifactsAreFresh } from "../crates/physics/scripts/kernel-inputs";

const root = resolve(import.meta.dir, "..");
if (!audioArtifactIsFresh(root)) await $`bun scripts/audio.ts`.cwd(root);
await $`bun scripts/tooling.ts`.cwd(root);
if (!kernelArtifactsAreFresh(root)) await $`bun crates/physics/scripts/build-kernel.ts`.cwd(root);
