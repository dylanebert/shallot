// Run one scene, capturing aligned b3Profile fields, hashes and symbolized main-thread self samples.
import { join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
import { nativeBinary, run } from "../native";
const root = process.argv[2];
const scene = process.argv[3] ?? "rain";
if (!root) throw new Error("usage: bun diagnostics/box3d-parity/kernel-gap/run.ts DIAGNOSTIC_ROOT rain|junkyard|joint_grid");
const windows: Record<string, [number, number]> = { rain: [280, 320], junkyard: [180, 200], joint_grid: [20, 40] };
const window = windows[scene];
if (!window) throw new Error(`unknown scene ${scene}`);
const [from, to] = window;
const evidence = process.env.EVIDENCE_DIR ?? import.meta.dir;
mkdirSync(evidence, { recursive: true });
const out = join(root, "diagnostic-bundle");
mkdirSync(out, { recursive: true });
const built = await Bun.build({ entrypoints: [join(root, "diagnostics/box3d-parity/scenes.ts")], outdir: out, target: "node", format: "esm" });
if (!built.success) throw new Error(built.logs.join("\n"));
const counted = Boolean(process.env.NATIVE_COUNTS);
const binary = process.env.NATIVE_COUNTS ?? nativeBinary();
for (const threads of [1, 4]) {
    const args = [scene, String(threads), String(to)];
    const n = run([binary, ...args], { PROFILE: String(from) });
    const s = run(["node", join(out, "scenes.js"), ...args], counted ? { PROFILE: String(from) } : { PROFILE: String(from), CPU: String(from) });
    const hashes = (text: string) => text.split("\n").filter(line => /^\d+ 0x/.test(line)).join("\n");
    if (hashes(n) !== hashes(s)) throw new Error(`${scene}/${threads}: hashes differ`);
    if (counted) {
        for (const prefix of ["D ", "G ", "P "]) {
            const rows = (text: string) => text.split("\n").filter(line => line.startsWith(prefix) && Number(line.split(" ")[1]) >= from).map(line => {
                if (prefix !== "D ") return line;
                const fields = line.split(" ");
                const kept = fields.slice(0,2);
                for (let i = 2; i < fields.length; i += 2)
                    if (!["clip_copy_points", "clip_zero_points", "tree_queries", "tree_nodes", "tree_leaves", "rebuild_leaves"].includes(fields[i])) kept.push(fields[i],fields[i+1]);
                return kept.join(" ");
            }).join("\n");
            if (rows(n) !== rows(s)) throw new Error(`${scene}/${threads}: ${prefix.trim()} counters differ`);
        }
    }
    writeFileSync(join(evidence, `${scene}-${threads}-${counted ? "counts-" : ""}native.txt`), n);
    writeFileSync(join(evidence, `${scene}-${threads}-${counted ? "counts-" : ""}kernel.txt`), s);
    console.log(`${scene}/${threads}: ${to} hashes equal; raw profiles written`);
}
