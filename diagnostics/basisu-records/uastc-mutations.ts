import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
const path = resolve(root, "crates/basisu/src/uastc.rs");
const original = readFileSync(path, "utf8");
const before = "        Ok(output)";
if (original.split(before).length !== 2) throw new Error("transcoder return must match once");
try {
    for (const [variant, target] of [["Astc", "astc"], ["Bc7", "bc7"], ["Rgba8", "rgba8"]]) {
        writeFileSync(path, original.replace(before, `        if matches!(target, UastcTarget::${variant}) { output[0] ^= 1; }\n${before}`));
        const result = spawnSync("cargo", ["test", "-p", "shallot-basisu", "--test", "ktx2", "every_uastc_target_is_bit_equal"], { cwd: root, encoding: "utf8" });
        writeFileSync(path, original);
        if (result.status === 0 || !result.stdout.includes("test result: FAILED") || !result.stdout.includes(`level 0 ${target}`)) throw new Error(`${target}: not the expected digest assertion red\n${result.stdout}\n${result.stderr}`);
        console.log(`RED UASTC ${target}: perturbed output caught by reference digest`);
    }
} finally {
    writeFileSync(path, original);
}
console.log("3 target mutants caught; source restored");
