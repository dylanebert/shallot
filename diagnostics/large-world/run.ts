import { copyFileSync, mkdtempSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { shallot } from "@dylanebert/shallot/bun";
import { removeTestDirectory } from "../../scripts/test-directory";
import { nativeBinary, run } from "../box3d-parity/native";

// bun diagnostics/large-world/run.ts [grid count, default 1000] [cpu|draw|shared] [workerCount: 1 shared, 0 unshared]
// At Box3D's grid of 1000, with BOX3D set, native.c runs the same scene first and each phase is
// reported as a ratio of Shallot to native.
const root = resolve(import.meta.dir, "../..");
const grid = Number(process.argv[2] ?? 1000);
type Line = { phase: string; milliseconds: number };
const lines = (text: string): Line[] =>
    text.split("\n").filter((line) => line.startsWith("{")).map((line) => JSON.parse(line));
let native: Line[] | undefined;
if (grid === 1000 && process.env.BOX3D) {
    const binary = nativeBinary("../large-world/native.c", undefined, join(process.env.BOX3D, "shared/benchmarks.c"));
    const output = run([binary]);
    process.stdout.write(output.replaceAll('{"phase"', '{"native":true,"phase"'));
    native = lines(output);
} else console.log(`no native side: it runs at grid 1000 with BOX3D set, not grid ${grid}${process.env.BOX3D ? "" : " without BOX3D"}`);
const dir = mkdtempSync(join(tmpdir(), "shallot-large-world-"));
try {
    const built = await Bun.build({
        entrypoints: [join(import.meta.dir, "cost.ts")],
        target: "node", format: "esm", outdir: dir, naming: "cost.mjs",
        external: ["webgpu"], plugins: [shallot({ root })],
    });
    if (!built.success) throw new Error(built.logs.map(String).join("\n"));
    symlinkSync(join(root, "node_modules"), join(dir, "node_modules"), "junction");
    copyFileSync(join(root, "src/extras/text/Inter-Regular.ttf"), join(dir, "Inter-Regular.ttf"));
    const child = Bun.spawn(["node", join(dir, "cost.mjs"), ...process.argv.slice(2)], { stdin: "inherit", stdout: "pipe", stderr: "inherit" });
    const output = await new Response(child.stdout).text();
    process.stdout.write(output);
    process.exitCode = await child.exited;
    if (native && process.exitCode === 0) {
        const shallotLines = lines(output);
        const ms = (from: Line[], phase: string) => from.find((line) => line.phase === phase)!.milliseconds;
        const pairs: [string, number, number][] = [
            ["author / create", ms(shallotLines, "author"), ms(native, "create")],
            ["first tick / first step", ms(shallotLines, "first tick (sync and solve)"), ms(native, "first step")],
            ["second tick / second step", ms(shallotLines, "second tick"), ms(native, "second step")],
        ];
        pairs.push(["author through second tick / create through second step", pairs.reduce((sum, pair) => sum + pair[1], 0), pairs.reduce((sum, pair) => sum + pair[2], 0)]);
        pairs.push(["20 hashes / native b3HashWorldState", ms(shallotLines, "20 hashes"), ms(native, "20 hashes")]);
        for (const [phase, shallotMs, nativeMs] of pairs)
            console.log(JSON.stringify({ ratio: phase, shallot: shallotMs, native: nativeMs, value: shallotMs / nativeMs }));
    }
} finally { removeTestDirectory(dir); }
