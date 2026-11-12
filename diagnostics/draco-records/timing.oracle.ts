import { resolve } from "node:path";
import { bunChild } from "../../scripts/bun-child";
import { timing } from "./comparison";
import { loadReference, readStreams, referenceDirectory, WASM } from "./reference";

const root = resolve(import.meta.dir, "../..");
referenceDirectory();
if (!process.argv.includes("--child")) {
    console.log(`Reference: Draco ${WASM} through draco_wasm_wrapper_gltf.js at the pin, as three.js DRACOLoader decodes`);
    for (let i = 0; i < 3; i++) {
        const result = Bun.spawnSync({ ...bunChild([import.meta.path, "--child", String(i)]), cwd: root, stdout: "inherit", stderr: "inherit" });
        if (result.exitCode !== 0) throw new Error(`timing process ${i + 1} failed`);
    }
} else {
    timing(await loadReference(), readStreams(), Number(process.argv.at(-1)));
}
