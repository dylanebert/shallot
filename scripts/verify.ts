import { resolve } from "node:path";
import type { Result } from "../packages/shallot-cli/bin/verify";

// Shared thin wrapper the repo bench/flows scripts drive the shipped gate through. `shallot verify` boots
// the target (an ejected vite app — the gym or a flow project), picks its own port, runs the published
// `window.__harness` in a real browser, and prints a JSON Result under `--json`. These scripts spawn it,
// pull that JSON off stdout, and interpret it — no server boot, no port logic, no Playwright here.

const repoRoot = resolve(import.meta.dir, "..");
export const REPO_ROOT = repoRoot;
export const CLI = resolve(repoRoot, "packages/shallot-cli/bin/cli.ts");

// The result types are `bin/verify.ts`'s own, re-exported rather than mirrored: this wrapper reads what
// that CLI printed, so a field it declares and a field this file declares can only ever drift. `Check`
// and `Verdict` come from the published harness protocol, the same source the CLI reads them from.
export type { Check, Verdict } from "@dylanebert/shallot/harness";
export type {
    CpuProfileBucket,
    CpuProfileEntry,
    CpuProfileSummary,
    LoAFEntry,
    LoAFScriptEntry,
    MemoryStats as Memory,
    RenderProbe,
} from "../packages/shallot-cli/bin/verify";

/** the `shallot verify --json` Result as a driver reads it back off stdout: `bin/verify.ts`'s own
 *  {@link Result}, every field optional because a setup failure emits `{ pass:false, error }` and nothing
 *  else, plus that `error` — the one field the envelope carries and the Result type does not. `artifacts`
 *  narrows to {@link ShaderArtifactSummary}: a driver reads the diagnostic, never the WGSL source. */
export type VerifyResult = Omit<Partial<Result>, "artifacts"> & {
    pass: boolean;
    error?: string;
    artifacts?: ShaderArtifactSummary[];
};

/** one captured shader record, as `bin/verify.ts` declares it on `Result.artifacts`. */
type ShaderArtifact = NonNullable<Result["artifacts"]>[number];

/** the diagnostic slice a driver surfaces when a red carries no console error: the record's identity, its
 *  compilation error and the human-readable parts of its messages. Narrower than {@link ShaderArtifact} on
 *  purpose — a driver never prints the WGSL source, hash or byte offsets — but every field's type is that
 *  record's, so the CLI stays the one definition. */
export type ShaderArtifactSummary = Pick<ShaderArtifact, "label" | "stage"> & {
    compilationError?: ShaderArtifact["compilationError"];
    messages?: Pick<
        ShaderArtifact["messages"][number],
        "type" | "message" | "lineNum" | "linePos"
    >[];
};

// verify drives a headed browser against local hardware, so its one prerequisite is a display: on Linux
// a session with neither DISPLAY nor WAYLAND_DISPLAY has no headed launch and therefore no conformant
// adapter (measured: headless Chrome falls back to a software rasterizer, which misses the device floor).
// Returns a human reason to skip, or null to proceed.
export function skipReason(): string | null {
    if (process.platform === "linux" && !(process.env.DISPLAY || process.env.WAYLAND_DISPLAY)) {
        return "no display";
    }
    return null;
}

/** `--query k=v` for each entry. */
export function queryFlags(query: string[]): string[] {
    return query.flatMap((q) => ["--query", q]);
}

/** find the JSON verify emits under --json: the last stdout line parsing to an object with a boolean
 *  `pass`. Vite/console chatter shares stdout, so scan for it rather than assume the last line. */
export function extractResult(stdout: string): VerifyResult | null {
    let found: VerifyResult | null = null;
    for (const line of stdout.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("{")) continue;
        try {
            const o = JSON.parse(t);
            if (o && typeof o.pass === "boolean") found = o as VerifyResult;
        } catch {
            // not the JSON line — keep scanning
        }
    }
    return found;
}

/** find the JSON array a batch (`--run`) verify emits under --json: the last stdout line parsing to an
 *  array. Same scan-don't-assume-last-line reasoning as {@link extractResult}. */
export function extractBatchResult(stdout: string): VerifyResult[] | null {
    let found: VerifyResult[] | null = null;
    for (const line of stdout.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("[")) continue;
        try {
            const o = JSON.parse(t);
            if (Array.isArray(o)) found = o as VerifyResult[];
        } catch {
            // not the JSON line — keep scanning
        }
    }
    return found;
}

// shared spawn: `shallot verify <dir> --json <extra>` from the repo root, stdout captured and (unless
// `quiet`) echoed. `verify`/`verifyBatch` differ only in how they parse the resulting stdout — a single
// object vs a JSON array — so the spawn itself has one source of truth. Returns the stdout and the
// process exit code; a nonzero exit must redden the verdict regardless of parsed stdout.
async function spawnVerify(
    dir: string,
    extra: string[],
    quiet: boolean,
): Promise<{ stdout: string; exitCode: number }> {
    const cmd = ["bun", CLI, "verify", dir, "--json", ...extra];
    // `env: { ...process.env }` is required because Bun.spawn does NOT propagate runtime
    // process.env changes to the child process — only pre-existing shell env vars are inherited by
    // default, so a driver that sets one for the verify run must pass the whole environment.
    const proc = Bun.spawn(cmd, {
        cwd: repoRoot,
        stdout: "pipe",
        stderr: "inherit",
        env: { ...process.env },
    });
    const stdout = await new Response(proc.stdout).text();
    if (!quiet) process.stdout.write(stdout);
    const exitCode = await proc.exited;
    return { stdout, exitCode };
}

/** Apply the exit-code gate to a parsed verify Result: a nonzero child exit reddens the verdict
 *  regardless of what stdout says — the guard the S1 fix added (the discarded-code fail-open).
 *  Exported so the S3 arm can exercise the decision with a real subprocess that prints a passing
 *  envelope and exits nonzero, without needing the full CLI + browser stack. */
export function applyExitCodeGate(
    result: VerifyResult | null,
    exitCode: number,
): VerifyResult | null {
    if (exitCode !== 0 && result?.pass === true) {
        return {
            ...result,
            pass: false,
            error: result.error ?? `verify process exited ${exitCode}`,
        };
    }
    if (exitCode !== 0 && result?.pass === false) {
        return {
            ...result,
            error: result.error ?? `verify process exited ${exitCode}`,
        };
    }
    if (exitCode !== 0 && !result) {
        return { pass: false, error: `verify process exited ${exitCode}` };
    }
    return result;
}

/** spawn `shallot verify <dir> --json <extra>` from the repo root and return the parsed Result (null if
 *  none was emitted — a crash before verify could report). Echoes verify's stdout so a single run shows
 *  its full envelope; `quiet` suppresses the echo for a many-cell sweep where the blobs drown the table. */
export async function verify(
    dir: string,
    extra: string[] = [],
    quiet = false,
): Promise<VerifyResult | null> {
    const { stdout, exitCode } = await spawnVerify(dir, extra, quiet);
    const result = applyExitCodeGate(extractResult(stdout), exitCode);
    if (!quiet) reportWarnings(result);
    return result;
}

/** print a run's non-fatal console warnings after its verdict, so `bun run flows` / `bun run recipes` /
 *  `bun bench` show the same noise a human sees in devtools — the typegpu `[implicit-conversion]`
 *  stream sat in every example's console for weeks while every wrapper here printed nothing. */
export function reportWarnings(result: VerifyResult | null): void {
    const warnings = result?.warnings ?? [];
    if (warnings.length === 0) return;
    const counts = new Map<string, number>();
    for (const w of warnings) {
        const line = w.split("\n")[0];
        counts.set(line, (counts.get(line) ?? 0) + 1);
    }
    console.log(`  ⚠ ${warnings.length} console warning(s):`);
    for (const [line, n] of counts) console.log(`    ${n > 1 ? `×${n} ` : ""}${line}`);
}

/** Page observations remain independent of process and protocol failures. Null slots are unavailable. */
export interface BatchOutcome {
    results: (VerifyResult | null)[];
    bytes: number;
    exitCode: number;
    errors: string[];
    pass: boolean;
}

/** Bind each observation to its requested query before a caller attributes a page verdict. */
export function batchOutcome(stdout: string, exitCode: number, runs: string[]): BatchOutcome {
    const parsed = extractBatchResult(stdout);
    const errors: string[] = [];
    let envelopes = 0;
    for (const line of stdout.split("\n")) {
        try {
            if (Array.isArray(JSON.parse(line))) envelopes++;
        } catch {
            /* console chatter */
        }
    }
    if (envelopes !== 1) errors.push(`batch envelopes: expected 1, received ${envelopes}`);
    if (new Set(runs).size !== runs.length) errors.push("batch requests: duplicate run");
    const results: (VerifyResult | null)[] = runs.map(() => null);
    if (!runs.length || !parsed || parsed.length !== runs.length) {
        errors.push(`batch population: requested ${runs.length}, received ${parsed?.length ?? 0}`);
    }
    const seen = new Set<string>();
    for (let i = 0; i < (parsed?.length ?? 0); i++) {
        const result = parsed![i];
        if (
            !result ||
            typeof result.pass !== "boolean" ||
            typeof result.url !== "string" ||
            (result.verdict != null &&
                (typeof result.verdict !== "object" ||
                    (result.verdict.checks != null &&
                        (!Array.isArray(result.verdict.checks) ||
                            result.verdict.checks.some(
                                (c) =>
                                    !c || typeof c.ok !== "boolean" || typeof c.name !== "string",
                            )))))
        ) {
            errors.push(`batch result ${i}: malformed verdict`);
            continue;
        }
        try {
            const url = new URL(result.url);
            const query = new URLSearchParams(runs[i]);
            if (
                i >= runs.length ||
                seen.has(url.href) ||
                [...query].some(
                    ([key, value]) =>
                        url.searchParams.getAll(key).length !== 1 ||
                        url.searchParams.get(key) !== value,
                )
            ) {
                errors.push(`batch result ${i}: duplicate or misbound URL`);
                continue;
            }
            seen.add(url.href);
            results[i] = result;
        } catch {
            errors.push(`batch result ${i}: invalid URL`);
        }
    }
    const failed = results.some(
        (r) => r && (!r.pass || r.verdict?.ok === false || r.verdict?.checks?.some((c) => !c.ok)),
    );
    if (exitCode !== 0) errors.push(`verify process exited ${exitCode}`);
    else if (failed) errors.push("batch protocol: failed page with successful process exit");
    return {
        results,
        bytes: Buffer.byteLength(stdout, "utf8"),
        exitCode,
        errors,
        pass: errors.length === 0 && results.length > 0 && results.every((r) => r?.pass) && !failed,
    };
}

/** spawn `shallot verify <dir> --json <extra> --run <r> --run <r> ...` — the shipped CLI's batch mode:
 *  one boot, one verdict per `runs` entry, JSON array out. Each
 *  `runs` entry is one `--run` spec (`"scenario=name"`, `&`-joined for more than one query key); `extra`
 *  carries the params shared across every run (`--query` flags, `--memory`). */
export async function verifyBatch(
    dir: string,
    runs: string[],
    extra: string[] = [],
    quiet = false,
): Promise<BatchOutcome> {
    const runFlags = runs.flatMap((r) => ["--run", r]);
    const { stdout, exitCode } = await spawnVerify(dir, [...extra, ...runFlags], quiet);
    return batchOutcome(stdout, exitCode, runs);
}

export interface VerifyCommandDeps {
    /** the CLI entrypoint to spawn — the repo's own by default; a test drives a stub through it. */
    cli?: string;
}

/** The command that runs a project's own verify gate: the local CLI, with the project's verify
 *  arguments passed through untouched. */
export function verifyCommand(argv: string[], deps: VerifyCommandDeps = {}): string[] {
    return ["bun", deps.cli ?? CLI, "verify", ...argv];
}
