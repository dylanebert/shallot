// Set up one eval task's project: pack the engine, scaffold a fresh project with `create-shallot`,
// install the packed tarball, and drop the task's PROMPT.md in. The project lands in an out-of-tree
// temp dir (os tmpdir, not under the repo) so the agent that works there sees only what ships on npm —
// `node_modules/@dylanebert/shallot` — and cannot read the engine source to cheat. The withheld gate
// stays in the repo; it is never copied into the project. Prints the project dir on the last line.
//
// `--bare` sets up the without-context arm of the shipped-context delta: the packed engine installs as
// usual, then the shipped `examples/` corpus (recipes + index) is removed from the installed package and
// the scaffold's agent docs lose the section pointing at it. The agent is left with the code, its JSDoc,
// and the product workflow (build/run/verify) — but none of the version-matched teaching context.
//
// Run: `bun run evals/setup.ts <task> [--json] [--bare]`  (or `bun run eval:setup <task>`)

import {
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const EVALS = import.meta.dir;
const ENGINE = resolve(EVALS, "../packages/shallot");
const CREATE_SHALLOT = resolve(EVALS, "../packages/create-shallot/index.ts");

function run(cmd: string[], cwd: string): { ok: boolean; out: string } {
    const p = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
    return { ok: p.exitCode === 0, out: `${p.stdout.toString()}\n${p.stderr.toString()}` };
}

function pack(dir: string, dest: string): string {
    mkdirSync(dest, { recursive: true });
    const r = run(["bun", "pm", "pack", "--destination", dest], dir);
    if (!r.ok) throw new Error(`pack ${dir} failed:\n${r.out}`);
    const tgz = readdirSync(dest).find((f) => f.endsWith(".tgz") && !f.startsWith("."));
    if (!tgz) throw new Error(`no tarball produced in ${dest}:\n${r.out}`);
    return join(dest, tgz);
}

// Strip the shipped `examples/` corpus out of the packed tarball itself (npm/bun pack lays every file
// under `package/`). The bare arm withholds the version-matched teaching context; deleting it only from
// the installed `node_modules` copy doesn't hold — `bun add` re-resolves the `file:` dep and re-extracts
// the tarball, so `examples/` comes back. Removing it at the tarball is the durable withholding. Untar →
// delete → re-tar, in place at the same path (package.json points a `file:` dep at it).
function stripTarball(tgz: string): void {
    const ex = mkdtempSync(join(tmpdir(), "shallot-eval-untar-"));
    const out = run(["tar", "-xzf", tgz, "-C", ex], ex);
    if (!out.ok) throw new Error(`untar ${tgz} failed:\n${out.out}`);
    rmSync(join(ex, "package/examples"), { recursive: true, force: true });
    const re = run(["tar", "-czf", tgz, "-C", ex, "package"], ex);
    if (!re.ok) throw new Error(`re-tar ${tgz} failed:\n${re.out}`);
    rmSync(ex, { recursive: true, force: true });
}

const args = process.argv.slice(2);
const asJson = args.includes("--json");
const bare = args.includes("--bare");
const task = args.find((a) => !a.startsWith("--"));
if (!task) {
    console.error("Usage: bun run evals/setup.ts <task> [--json] [--bare]");
    process.exit(1);
}

// Strip the "## Engine reference" section from the scaffold's agent docs — its only content is the
// pointers at the shipped `node_modules/.../AGENTS.md` + `examples/` docs, the context the bare arm
// withholds. The build/run/verify section (the shipped verify CLI is product surface, not context) and
// every other section stay.
function stripShippedContext(md: string): string {
    // anchor the end on the next section OR end-of-string, so the strip still fires if "Engine
    // reference" is ever the last section — a bare `(?=\n## )` would silently no-op and leak the context.
    return md.replace(/\n## Engine reference\n[\s\S]*?(?=\n## |$)/, "");
}
const taskDir = resolve(EVALS, "tasks", task);
const promptPath = join(taskDir, "PROMPT.md");
try {
    readFileSync(promptPath);
} catch {
    console.error(`no such task: ${task} (missing ${promptPath})`);
    process.exit(1);
}

// realpath: macOS tmpdir is a symlink; vite's fs.allow prefix check needs the resolved form
const work = realpathSync(mkdtempSync(join(tmpdir(), `shallot-eval-${task}-`)));
const proj = join(work, "app");

const engineTgz = pack(ENGINE, join(work, "engine-pack"));
// bare arm: strip the shipped context at the tarball so a later `bun add` can't re-extract it back
if (bare) stripTarball(engineTgz);

const scaffold = run(["bun", CREATE_SHALLOT, "app"], work);
if (!scaffold.ok) throw new Error(`create-shallot failed:\n${scaffold.out}`);

// a real user installs the published engine; the packed tarball stands in for it
const pkg = JSON.parse(readFileSync(join(proj, "package.json"), "utf8"));
pkg.dependencies["@dylanebert/shallot"] = `file:${engineTgz}`;
writeFileSync(join(proj, "package.json"), `${JSON.stringify(pkg, null, 2)}\n`);

const install = run(["bun", "install"], proj);
if (!install.ok) throw new Error(`bun install failed:\n${install.out.slice(-800)}`);

if (bare) {
    // examples/ is already gone from the tarball (stripTarball above); the scaffold docs still carry the
    // pointer section at it, so strip that here — reinstall-proof, unlike a node_modules deletion.
    for (const doc of ["AGENTS.md", "CLAUDE.md"]) {
        const path = join(proj, doc);
        writeFileSync(path, stripShippedContext(readFileSync(path, "utf8")));
    }
}

writeFileSync(join(proj, "PROMPT.md"), readFileSync(promptPath));
writeFileSync(
    join(proj, ".eval.json"),
    `${JSON.stringify({ task, bare, created: new Date().toISOString() }, null, 2)}\n`,
);

if (asJson) {
    console.log(JSON.stringify({ task, project: proj, work }));
} else {
    console.error(`task ${task}: project ready. Agent works with cwd = the path below.`);
    console.log(proj);
}
