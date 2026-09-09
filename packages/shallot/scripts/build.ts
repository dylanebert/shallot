import { resolve } from "node:path";
import { $ } from "bun";

await $`bun run build`.cwd(resolve(import.meta.dir, "../../shallot-runtime"));
await $`cargo build --release`.cwd(resolve(import.meta.dir, "../../shallot-cli/rust/window"));
