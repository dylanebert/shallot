import { resolve } from "node:path";
import { $ } from "bun";

await $`bun scripts/audio.ts`.cwd(resolve(import.meta.dir, ".."));
await $`bun scripts/tooling.ts`.cwd(resolve(import.meta.dir, ".."));
await $`bun scripts/physics/build-kernel.ts`.cwd(resolve(import.meta.dir, ".."));
