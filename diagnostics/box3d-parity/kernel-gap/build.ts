// Diagnostic build in an isolated archive: never overwrites shipping kernels.
import { mkdtempSync, readFileSync, writeFileSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "../../..");
const revision = process.argv[2] ?? "HEAD";
const dir = mkdtempSync(join(tmpdir(), "shallot-symbols-"));
function run(command: string[], cwd = dir) {
    const p = Bun.spawnSync(command, { cwd, stdout: "inherit", stderr: "inherit", env: { ...process.env, CARGO_PROFILE_RELEASE_STRIP: "false", CARGO_PROFILE_RELEASE_DEBUG: "line-tables-only" } });
    if (p.exitCode) throw new Error(`${command.join(" ")}: ${p.exitCode}`);
}
run(["bash", "-c", 'git archive "$1" | tar -x -C "$2"', "archive", revision, dir], root);
symlinkSync(join(root, "node_modules"), join(dir, "node_modules"));
const optimizer = join(dir, "scripts/wasm-opt.ts");
writeFileSync(optimizer, readFileSync(optimizer, "utf8").replace('"-O3",', '"-O3", "--debuginfo",'));
const cpu = join(dir, "diagnostics/box3d-parity/cpu.ts");
let source = readFileSync(cpu, "utf8");
source = source.replace('if (id === 7) {', `if (id === 0) {
            const sectionName = str();
            if (sectionName === "name") {
                while (at < end) {
                    const sub = b[at++], length = leb(), subEnd = at + length;
                    if (sub === 1) for (let n = leb(); n > 0; --n) {
                        const index = leb(); names.set(index, str());
                    }
                    at = subEnd;
                }
            }
        }
        if (id === 7) {`);
source = source.replace('const lines: string[] = [];', `const lines: string[] = [];
    const selfTimes = new Map<string, number>();
    const inclusiveTimes = new Map<string, number>();
    for (const [id, us] of time) {
        const frame = byId.get(id)?.callFrame;
        let inStep = false;
        for (let at: number | undefined = id; at !== undefined; at = parent.get(at))
            if (name(at) === "step") inStep = true;
        if (!inStep) continue;
        if (!frame?.url.startsWith("wasm")) continue;
        const key = name(id);
        selfTimes.set(key, (selfTimes.get(key) ?? 0) + us / 1000 / steps);
        const seen = new Set<string>();
        for (let at: number | undefined = id; at !== undefined; at = parent.get(at)) {
            if (!byId.get(at)?.callFrame.url.startsWith("wasm")) continue;
            const fn = name(at);
            if (seen.has(fn)) continue;
            seen.add(fn);
            inclusiveTimes.set(fn, (inclusiveTimes.get(fn) ?? 0) + us / 1000 / steps);
        }
    }
    for (const [key, ms] of [...inclusiveTimes].sort((a,b) => b[1]-a[1]))
        lines.push(\`KI \${ms.toFixed(6)} \${key}\`);
    for (const [key, ms] of [...selfTimes].sort((a,b) => b[1]-a[1]))
        lines.push(\`KS \${ms.toFixed(6)} \${key}\`);`);
writeFileSync(cpu, source);
run(["python3", join(import.meta.dir, "workers.py"), dir]);
if (process.argv.includes("--counts")) run(["python3", join(import.meta.dir, "instrument.py"), dir]);
run(["bun", "run", "crates/physics/scripts/build-kernel.ts"]);
console.log(`DIAGNOSTIC_ROOT=${dir}`);
