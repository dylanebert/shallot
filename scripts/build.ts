import { resolve } from "node:path";
import { $ } from "bun";
import { audioArtifactIsFresh } from "../crates/audio/scripts/audio-inputs";
import { basisuArtifactIsFresh } from "../crates/basisu/scripts/inputs";
import { dracoArtifactIsFresh } from "../crates/draco/scripts/inputs";
import { gltfArtifactIsFresh } from "../crates/gltf/scripts/inputs";
import { mikktspaceArtifactIsFresh } from "../crates/mikktspace/scripts/inputs";
import { kernelArtifactsAreFresh } from "../crates/physics/scripts/kernel-inputs";

const root = resolve(import.meta.dir, "..");
if (!audioArtifactIsFresh(root)) await $`bun scripts/audio.ts`.cwd(root);
if (!basisuArtifactIsFresh(root)) await $`bun crates/basisu/scripts/build.ts`.cwd(root);
if (!dracoArtifactIsFresh(root)) await $`bun crates/draco/scripts/build.ts`.cwd(root);
if (!gltfArtifactIsFresh(root)) await $`bun crates/gltf/scripts/build.ts`.cwd(root);
if (!mikktspaceArtifactIsFresh(root)) await $`bun crates/mikktspace/scripts/build.ts`.cwd(root);
await $`bun scripts/tooling.ts`.cwd(root);
if (!kernelArtifactsAreFresh(root)) await $`bun crates/physics/scripts/build-kernel.ts`.cwd(root);
