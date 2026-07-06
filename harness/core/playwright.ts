import { isWSL, stageOnWindows, type WindowsPaths } from "./wsl";

// The one place that runs `playwright test`. Native it spawns directly; under WSL it stages the
// launcher's config + test files onto the Windows host and drives them through powershell, so the
// host's real-GPU Chrome runs them. Both launchers depend on this — the gym (page.ts → a verdict on
// stdout) and capture (flows → screenshots on disk). The WSL staging and the powershell spawn, the
// most fragile code in the repo, live here once. The caller owns its config + tests and reads back
// its own artifacts (the gym decodes `stdout`; capture copies screenshots from `staged.wsl`).

export interface RunArgs {
    /** dir holding the playwright config + test files — the launcher's own directory */
    dir: string;
    /** config filename, relative to `dir` */
    config: string;
    /** trailing `playwright test` args (a positional test file, `--grep <name>`) */
    args?: string[];
    /** WSL staging: a temp-dir name + the files (relative to `dir`) copied to the host to run from there */
    stage?: { name: string; files: string[] };
    /** env for the run, built from the staged paths (`null` when native) so output dirs resolve host-side */
    env?: (staged: WindowsPaths | null) => Record<string, string>;
    /** true: stream the child's stdout (the live list reporter); false (default): capture + return it */
    inherit?: boolean;
    /** hard ceiling on the whole spawn — a backstop above Playwright's own `globalTimeout`, never the
     *  primary guard. Per-test / action timeouts in the config catch a hung flow long before this. */
    timeoutMs: number;
}

export interface RunResult {
    /** child exit code; `null` when the spawn ceiling killed it (the backstop fired) */
    exitCode: number | null;
    /** captured stdout (empty when `inherit`) */
    stdout: string;
    /** the spawn ceiling fired — distinct from a clean nonzero Playwright exit */
    timedOut: boolean;
    /** the Windows staging paths (WSL only, else `null`) — read artifacts back from `.wsl` */
    staged: WindowsPaths | null;
}

function quote(s: string): string {
    return s.replace(/'/g, "''");
}

function decode(stdout: Uint8Array | null | undefined, inherit: boolean): string {
    if (inherit) return "";
    const out = new TextDecoder().decode(stdout ?? new Uint8Array());
    process.stdout.write(out);
    return out;
}

export function runPlaywright(run: RunArgs): RunResult {
    return isWSL ? runWSL(run) : runNative(run);
}

function runNative(run: RunArgs): RunResult {
    const env = run.env?.(null) ?? {};
    const result = Bun.spawnSync(
        ["bunx", "playwright", "test", "--config", run.config, ...(run.args ?? [])],
        {
            cwd: run.dir,
            stdout: run.inherit ? "inherit" : "pipe",
            stderr: "inherit",
            timeout: run.timeoutMs,
            env: { ...process.env, ...env },
        },
    );
    return {
        exitCode: result.exitCode,
        stdout: decode(result.stdout, !!run.inherit),
        timedOut: result.exitCode === null,
        staged: null,
    };
}

function runWSL(run: RunArgs): RunResult {
    if (!run.stage) throw new Error("WSL playwright run needs a `stage` (name + files)");
    const staged = stageOnWindows(run.dir, run.stage.name, run.stage.files);
    const env = run.env?.(staged) ?? {};
    const assigns = Object.entries(env)
        .map(([k, v]) => `$env:${k} = '${quote(v)}';`)
        .join(" ");
    const tail = [run.config, ...(run.args ?? [])].map((a) => `'${quote(a)}'`).join(" ");
    const result = Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `${assigns} $env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\\ms-playwright"; cd '${staged.win}'; bunx playwright test --config ${tail}`,
        ],
        {
            stdout: run.inherit ? "inherit" : "pipe",
            stderr: "inherit",
            timeout: run.timeoutMs,
        },
    );
    return {
        exitCode: result.exitCode,
        stdout: decode(result.stdout, !!run.inherit),
        timedOut: result.exitCode === null,
        staged,
    };
}
