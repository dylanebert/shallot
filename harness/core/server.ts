import { basename, resolve } from "path";

const repoRoot = resolve(import.meta.dir, "..", "..");

/** Resolve a shallot example name to its project dir (`examples/<example>`). */
export function exampleDir(example: string): string {
    return resolve(repoRoot, "examples", example);
}

// total readiness budget; past it a non-answering server is wedged, not slow (override per machine via
// SERVER_STARTUP_TIMEOUT_MS).
const STARTUP_TIMEOUT_MS = Number(process.env.SERVER_STARTUP_TIMEOUT_MS) || 60_000;
// an unbounded `fetch` against a server that accepts the connection but never responds (a dev server mid
// first-bundle, or wedged) blocks forever, defeating the budget below; every probe is time-bounded so the
// budget is the only authority on how long boot waits.
const PROBE_TIMEOUT_MS = 2000;

// resolve true iff `url` answers within `PROBE_TIMEOUT_MS`; never throws, never hangs
async function answers(url: string): Promise<boolean> {
    try {
        await fetch(url, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
        return true;
    } catch {
        return false;
    }
}

// Boot a dev server on `port` and resolve once it answers. `cwd` + `cmd` is any project that serves a
// page: by default `bun run dev` in `cwd` (a shallot example, or an external workspace driving the
// window.__harness contract — the orrstead game bench); pass `cmd` to run something else from `cwd`
// (capture boots the editor via the shallot CLI against an in-repo fixture). `label` names it in
// log + error lines. `BROWSER=none` suppresses vite's auto-open — a harness never wants a tab.
//
// A crash or a wedged-but-listening server throws (naming the server + port, killing the child) within
// STARTUP_TIMEOUT_MS — the contract callers rely on to fail a run loud rather than hang it.
export async function startServer(
    cwd: string,
    port: number,
    label = basename(cwd),
    cmd: string[] = ["bun", "run", "dev", "--port", String(port), "--strictPort"],
): Promise<ReturnType<typeof Bun.spawn>> {
    const url = `http://localhost:${port}`;
    // free a stale holder of the port before spawning
    if (await answers(url)) {
        Bun.spawnSync(["fuser", "-k", `${port}/tcp`], { stdout: "ignore", stderr: "ignore" });
        await Bun.sleep(500);
    }

    const proc = Bun.spawn(cmd, {
        cwd,
        stdout: "ignore",
        stderr: "pipe",
        env: { ...process.env, BROWSER: "none" },
    });

    const deadline = Date.now() + STARTUP_TIMEOUT_MS;
    while (Date.now() < deadline) {
        if (proc.exitCode !== null) {
            const stderr = await new Response(proc.stderr).text();
            throw new Error(`${label} dev server exited early (code ${proc.exitCode}): ${stderr}`);
        }
        if (await answers(url)) {
            console.log(`${label} server ready on port ${port}`);
            return proc;
        }
        await Bun.sleep(500);
    }
    proc.kill();
    throw new Error(
        `${label} server did not answer on ${url} within ${STARTUP_TIMEOUT_MS / 1000}s — wedged or mid-bundle; aborting (raise SERVER_STARTUP_TIMEOUT_MS if it just needs longer)`,
    );
}
