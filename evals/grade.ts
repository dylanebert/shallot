// Grade one eval task's project: typecheck it, build it, then (with a display) boot it and drive the
// withheld gate against the running canvas. Emits a machine-readable result plus a human summary. The
// gate scripts live in the repo and are never shown to the agent — this is the only place they run.
//
// Uses evals' own browser path (evals/harness: server boot, WSL→Windows Playwright staging, display
// gating) — self-contained, so the repo's shipped-gate dissolution doesn't reach it. On the pre-agent
// (empty scaffold) project the gate is expected to FAIL its assertions — what this proves is the
// mechanics: build, boot, drive, report.
//
// Run: `bun run evals/grade.ts <task> <projectDir> [--json] [--port <n>]`

import {
    cpSync,
    existsSync,
    mkdtempSync,
    readFileSync,
    realpathSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { runPlaywright } from "./harness/playwright";
import { startServer } from "./harness/server";
import { detectDisplay, isWSL } from "./harness/wsl";

const EVALS = import.meta.dir;
const HARNESS = join(EVALS, "harness");
const RESULT_RE = /__EVAL_RESULT__([\s\S]+?)__EVAL_RESULT__/;

interface Assertion {
    name: string;
    ok: boolean;
    detail?: string;
}
interface GateEnvelope {
    ok: boolean;
    booted: boolean;
    rendered: boolean;
    assertions: Assertion[];
    errors?: string[];
}
interface Check {
    ok: boolean | null;
    detail?: string;
}
interface Result {
    task: string;
    project: string;
    timestamp: string;
    checks: {
        typecheck: Check;
        build: Check;
        gate: Check & { skipped?: boolean; assertions?: Assertion[]; errors?: string[] };
    };
    // mechanical fields this script fills; the judgment fields (claimed*/verified*) are filled by the
    // kex-side runner that reads the agent's transcript — verification honesty is a judgment, not a
    // mechanical check
    verification: {
        booted: boolean | null;
        rendered: boolean | null;
        claimedVerified: boolean | null;
        verifiedHonestly: boolean | null;
        notes: string | null;
    };
    pass: boolean | null;
}

function sh(cmd: string[], cwd: string): { ok: boolean; out: string } {
    const p = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
    return { ok: p.exitCode === 0, out: `${p.stdout.toString()}\n${p.stderr.toString()}` };
}

const args = process.argv.slice(2);
const asJson = args.includes("--json");
const portIdx = args.indexOf("--port");
const port = portIdx !== -1 ? Number(args[portIdx + 1]) : 5250;
const positional = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--port");
const [task, projectArg] = positional;
if (!task || !projectArg) {
    console.error("Usage: bun run evals/grade.ts <task> <projectDir> [--json] [--port <n>]");
    process.exit(1);
}
const project = resolve(projectArg);
const taskGate = join(EVALS, "tasks", task, "gate.ts");
if (!existsSync(taskGate)) throw new Error(`no gate for task ${task} at ${taskGate}`);
if (!existsSync(join(project, "package.json"))) throw new Error(`no project at ${project}`);

const CLI = join(project, "node_modules/@dylanebert/shallot/bin/cli.ts");
const result: Result = {
    task,
    project,
    timestamp: new Date().toISOString(),
    checks: { typecheck: { ok: null }, build: { ok: null }, gate: { ok: null } },
    verification: {
        booted: null,
        rendered: null,
        claimedVerified: null,
        verifiedHonestly: null,
        notes: null,
    },
    pass: null,
};

// typecheck — the check the scaffold's AGENTS.md tells the agent to run
const tc = sh(["bunx", "tsc", "--noEmit"], project);
result.checks.typecheck = { ok: tc.ok, detail: tc.ok ? undefined : tc.out.trim().slice(-600) };

// build — the shipped CLI, the way an installed user ships the project
const build = sh(["bun", CLI, "build", "."], project);
const built = build.ok && existsSync(join(project, "dist", "index.html"));
result.checks.build = {
    ok: built,
    detail: built ? undefined : build.out.trim().slice(-800) || "no dist/index.html",
};

// gate — boot the project and drive the withheld assertions in a real browser
if (!detectDisplay()) {
    result.checks.gate = { ok: null, skipped: true, detail: "no display — browser gate skipped" };
} else {
    const runDir = realpathSync(mkdtempSync(join(tmpdir(), `shallot-eval-gate-${task}-`)));
    cpSync(join(HARNESS, "package.json"), join(runDir, "package.json"));
    cpSync(join(HARNESS, "gate.config.ts"), join(runDir, "gate.config.ts"));
    cpSync(join(HARNESS, "lib.ts"), join(runDir, "lib.ts"));
    // flatten the gate's import so it sits beside lib.ts in the staged run dir
    const src = readFileSync(taskGate, "utf8").replaceAll('"../../harness/lib"', '"./lib"');
    writeFileSync(join(runDir, "gate.ts"), src);

    // native (non-WSL) runs Playwright in-place, so install its deps there; WSL stages + installs
    // host-side inside runPlaywright
    if (!isWSL) {
        sh(["bun", "install"], runDir);
        sh(["bunx", "playwright", "install", "chromium"], runDir);
    }

    const url = `http://localhost:${port}/`;
    const server = await startServer(project, port, `eval-${task}`, [
        "bun",
        CLI,
        "dev",
        ".",
        "--port",
        String(port),
        "--strict-port",
    ]);
    try {
        const run = runPlaywright({
            dir: runDir,
            config: "gate.config.ts",
            args: ["gate.ts"],
            stage: {
                name: `shallot-eval-gate`,
                files: ["package.json", "gate.config.ts", "lib.ts", "gate.ts"],
            },
            env: () => ({ EVAL_URL: url }),
            timeoutMs: 240_000,
        });
        const m = run.stdout.match(RESULT_RE);
        if (m) {
            const env = JSON.parse(m[1]) as GateEnvelope;
            result.checks.gate = {
                ok: env.ok,
                assertions: env.assertions,
                errors: env.errors,
                detail: env.booted ? undefined : "project did not boot a canvas",
            };
            result.verification.booted = env.booted;
            result.verification.rendered = env.rendered;
        } else {
            result.checks.gate = {
                ok: false,
                detail: `gate produced no result (exit ${run.exitCode})`,
            };
        }
    } finally {
        server.kill();
    }
}

const g = result.checks.gate;
result.pass =
    g.skipped || g.ok === null
        ? null
        : result.checks.typecheck.ok === true && result.checks.build.ok === true && g.ok === true;

if (asJson) {
    console.log(JSON.stringify(result, null, 2));
} else {
    const mark = (c: boolean | null) => (c === null ? "–" : c ? "✓" : "✗");
    console.log(`\neval: ${task}  (${project})`);
    console.log(`  ${mark(result.checks.typecheck.ok)} typecheck`);
    console.log(`  ${mark(result.checks.build.ok)} build`);
    if (g.skipped) console.log(`  – gate skipped (no display)`);
    else {
        console.log(
            `  ${mark(g.ok)} gate  (booted ${g.ok === null ? "?" : result.verification.booted}, rendered ${result.verification.rendered})`,
        );
        for (const x of g.assertions ?? []) {
            console.log(`      ${x.ok ? "✓" : "✗"} ${x.name}${x.detail ? ` — ${x.detail}` : ""}`);
        }
        if (g.errors?.length) console.log(`      errors: ${g.errors.slice(0, 3).join(" | ")}`);
    }
    const verdict =
        result.pass === null ? "INCOMPLETE (gate did not run)" : result.pass ? "PASS" : "FAIL";
    console.log(`  => ${verdict}`);
    console.log(`\n${JSON.stringify(result)}`);
}
