import { resolve } from "node:path";
import { bunChild } from "../../scripts/bun-child";
import { timing } from "./comparison";
import { readInputs } from "./controls";
import { loadReference, referenceDirectory } from "./reference";

const root = resolve(import.meta.dir, "../..");
referenceDirectory();
if (!process.argv.includes("--child")) {
    const sync = Bun.spawnSync({ ...bunChild([resolve(import.meta.dir, "records.ts")]), cwd: root, stdout: "pipe", stderr: "pipe" });
    if (sync.exitCode !== 0) throw new Error(new TextDecoder().decode(sync.stderr));
    console.log("Reference: Binomial webgl/transcoder/build/basis_transcoder.{js,wasm}; 32c7b74 built 3784d52; transcoder inputs unchanged at 073eb6a0");
    for (let i = 0; i < 3; i++) {
        const result = Bun.spawnSync({ ...bunChild([import.meta.path, "--child", String(i)]), cwd: root, stdout: "inherit", stderr: "inherit" });
        if (result.exitCode !== 0) throw new Error(`timing process ${i + 1} failed`);
    }
} else {
    const module = await loadReference();
    timing(module, readInputs(), Number(process.argv.at(-1)));
}
