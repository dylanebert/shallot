// The one place that runs `playwright test`. It spawns the runner locally, headed on the seat's own
// display (`grade.ts` refuses the gate outright when there is none), and returns the child's stdout —
// where the gate's result envelope is emitted — with its exit code. The caller owns its config + test
// files and reads back its own artifacts (grade.ts decodes `stdout`).

export interface RunArgs {
    /** dir holding the playwright config + test files — the launcher's own directory */
    dir: string;
    /** config filename, relative to `dir` */
    config: string;
    /** trailing `playwright test` args (a positional test file, `--grep <name>`) */
    args?: string[];
    /** env for the run */
    env?: () => Record<string, string>;
    /** true: stream the child's stdout (the live list reporter); false (default): capture + return it */
    inherit?: boolean;
    /** hard ceiling on the whole spawn — a backstop above `gate.config.ts`'s own `globalTimeout`
     *  (itself above every gate's own `test.setTimeout`), each level derived from `harness/lib`'s one
     *  owner rather than a separately hand-picked number, so the three stay ordered as the owner
     *  moves. Not a guard sized with comfortable slack by assumption: the worst gate's own per-test
     *  budget is the largest term the ceilings above it are built from. */
    timeoutMs: number;
}

export interface RunResult {
    /** child exit code; `null` when the spawn ceiling killed it (the backstop fired) */
    exitCode: number | null;
    /** captured stdout (empty when `inherit`) */
    stdout: string;
    /** the spawn ceiling fired — distinct from a clean nonzero Playwright exit */
    timedOut: boolean;
}

function decode(stdout: Uint8Array | null | undefined, inherit: boolean): string {
    if (inherit) return "";
    const out = new TextDecoder().decode(stdout ?? new Uint8Array());
    process.stdout.write(out);
    return out;
}

export function runPlaywright(run: RunArgs): RunResult {
    const env = run.env?.() ?? {};
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
    };
}
