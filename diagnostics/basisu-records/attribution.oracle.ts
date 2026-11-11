import { resolve } from "node:path";
import { bunChild } from "../../scripts/bun-child";
import { attribution } from "./attribution";
import { controls } from "./controls";
import { loadReference } from "./reference";

const root = resolve(import.meta.dir, "../..");
const sync = Bun.spawnSync({ ...bunChild([resolve(import.meta.dir, "records.ts")]), cwd: root, stdout: "pipe", stderr: "pipe" });
if (sync.exitCode !== 0) throw new Error(new TextDecoder().decode(sync.stderr));
attribution(await loadReference(), controls());
