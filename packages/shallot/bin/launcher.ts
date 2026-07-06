// PROVISIONAL — a per-project-vite concretization of the project runtime, kept while the editor's
// distribution layering settles (see exports.md "Distribution layers"). Not yet wired into a shipping
// path: the convenience bundle composes the canonical tool path and may harvest only this file's
// recents registry. Don't treat it as settled foundation.
//
// The launcher backend: a persistent seam that owns project-server lifecycle and a recent-projects
// registry. It spawns/kills one per-project editor dev server per open project (the `shallot dev`
// shape, isolated as its own OS process), driven over `runSidecar`'s stdin/stdout control protocol.
// vite-free by construction — the editor launch is a command string array, never an import — so the
// lifecycle is testable on its own, no GPU/ECS/vite-internals.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { basename, dirname, join, resolve } from "node:path";

const REGISTRY_VERSION = 1;
const MAX_RECENTS = 20;

/** a recent project as persisted: its absolute path, display name, and last-opened ISO timestamp. */
export interface RecentProject {
    path: string;
    name: string;
    opened: string;
}

/** {@link RecentProject} annotated for display — `exists` lets the project list grey a moved-away dir. */
export interface RecentEntry extends RecentProject {
    exists: boolean;
}

/** the on-disk recents registry, most-recent-first. */
export interface Registry {
    version: number;
    projects: RecentProject[];
}

/** a project's running editor server: the dir it serves, its port, and the URL the shell navigates to. */
export interface OpenProject {
    dir: string;
    port: number;
    url: string;
}

/** builds the child command that serves `dir`'s editor on `port` — the seam tests stub to skip vite. */
export type Launch = (dir: string, port: number) => string[];

export interface LauncherOptions {
    /** registry file override (tests point this at a tmp file; production uses {@link registryPath}). */
    registry?: string;
    /** child-command override (tests stub a vite-free server; production launches the editor). */
    command?: Launch;
}

type Proc = ReturnType<typeof Bun.spawn>;

/** per-user shallot data dir, mirroring the native window crate's `cache_dir` so both agree on location. */
export function dataDir(): string {
    if (process.platform === "win32") {
        const local =
            process.env.LOCALAPPDATA ?? join(process.env.USERPROFILE ?? ".", "AppData", "Local");
        return join(local, "shallot");
    }
    if (process.platform === "darwin") {
        return join(process.env.HOME ?? ".", "Library", "Application Support", "shallot");
    }
    const base = process.env.XDG_DATA_HOME ?? join(process.env.HOME ?? ".", ".local", "share");
    return join(base, "shallot");
}

/** the recents registry file: `<dataDir>/launcher/projects.json`. */
export function registryPath(): string {
    return join(dataDir(), "launcher", "projects.json");
}

function isRecent(v: unknown): v is RecentProject {
    const r = v as RecentProject;
    return (
        typeof v === "object" &&
        v !== null &&
        typeof r.path === "string" &&
        typeof r.name === "string" &&
        typeof r.opened === "string"
    );
}

/** read the recents registry, tolerating an absent or corrupt file (a first run, a hand-edit). */
export function loadRegistry(file = registryPath()): Registry {
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(file, "utf-8"));
    } catch {
        return { version: REGISTRY_VERSION, projects: [] };
    }
    const obj = parsed as { projects?: unknown };
    const projects = Array.isArray(obj?.projects) ? obj.projects.filter(isRecent) : [];
    return { version: REGISTRY_VERSION, projects };
}

/** persist the registry, creating the parent dir on first write. */
export function saveRegistry(reg: Registry, file = registryPath()): void {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(reg, null, 2) + "\n");
}

/** upsert a project to the front (most-recent-first), deduped by absolute path, capped at {@link MAX_RECENTS}. */
export function addRecent(reg: Registry, dir: string, now: string): Registry {
    const path = resolve(dir);
    const entry: RecentProject = { path, name: basename(path), opened: now };
    const rest = reg.projects.filter((p) => p.path !== path);
    return { version: REGISTRY_VERSION, projects: [entry, ...rest].slice(0, MAX_RECENTS) };
}

/** drop a project from the registry by absolute path. */
export function removeRecent(reg: Registry, dir: string): Registry {
    const path = resolve(dir);
    return { version: REGISTRY_VERSION, projects: reg.projects.filter((p) => p.path !== path) };
}

// the editor dev server for `dir` on `port` — the exact shape `scripts/capture.ts` spawns and proves
// green. `--strict-port` makes a port collision fail the child fast; without it vite drifts to another
// port the launcher never learns, so waitReady would hang polling the dead one.
const CLI = resolve(import.meta.dir, "cli.ts");
function editorCommand(dir: string, port: number): string[] {
    return ["bun", CLI, "edit", dir, "--port", String(port), "--strict-port"];
}

/** an ephemeral free TCP port: bind `:0`, read the assigned port, release it for the child to claim. */
function freePort(): Promise<number> {
    return new Promise((res, rej) => {
        const srv = createServer();
        srv.once("error", rej);
        srv.listen(0, () => {
            const addr = srv.address();
            const port = typeof addr === "object" && addr ? addr.port : 0;
            srv.close(() => res(port));
        });
    });
}

// resolve once the server answers (the condition), or throw if the child exited early (surfacing its
// stderr). The 200ms between polls is the interval, not a fixed wait — the loop exits on the answer. The
// ~30s budget matches harness/core/server.ts: a cold editor vite boot (svelte + optimizeDeps) is slow.
async function waitReady(port: number, proc: Proc, label: string): Promise<void> {
    for (let i = 0; i < 150; i++) {
        if (proc.exitCode !== null) {
            const stderr = await new Response(proc.stderr as ReadableStream).text();
            throw new Error(`${label} server exited early (code ${proc.exitCode}): ${stderr}`);
        }
        try {
            await fetch(`http://localhost:${port}`);
            return;
        } catch {
            await Bun.sleep(200);
        }
    }
    throw new Error(`${label} server failed to start on port ${port}`);
}

/**
 * the project-server lifecycle, holding the running set keyed by absolute dir. The persistent backend
 * the native shell hosts: it opens a project (spawns its editor server, records it in recents) and
 * closes one (kills the server, releases the port). "No project open" is the empty set; {@link recents}
 * is the front-door list. `registry` + `command` are injectable so the lifecycle tests run vite-free.
 */
export function createLauncher(opts: LauncherOptions = {}) {
    const file = opts.registry ?? registryPath();
    const command = opts.command ?? editorCommand;
    const open = new Map<string, { project: OpenProject; proc: Proc }>();

    function recents(): RecentEntry[] {
        return loadRegistry(file).projects.map((p) => ({ ...p, exists: existsSync(p.path) }));
    }

    function isOpen(dir: string): boolean {
        return open.has(resolve(dir));
    }

    async function openProject(dir: string): Promise<OpenProject> {
        const path = resolve(dir);
        const running = open.get(path);
        if (running) return running.project;

        const port = await freePort();
        const proc = Bun.spawn(command(path, port), {
            env: { ...process.env, BROWSER: "none" },
            stdout: "ignore",
            stderr: "pipe",
        });
        await waitReady(port, proc, basename(path));

        const project: OpenProject = { dir: path, port, url: `http://localhost:${port}` };
        open.set(path, { project, proc });
        saveRegistry(addRecent(loadRegistry(file), path, new Date().toISOString()), file);
        return project;
    }

    async function closeProject(dir: string): Promise<void> {
        const path = resolve(dir);
        const running = open.get(path);
        if (!running) return;
        running.proc.kill();
        await running.proc.exited;
        open.delete(path);
    }

    async function closeAll(): Promise<void> {
        await Promise.all([...open.keys()].map(closeProject));
    }

    return { recents, isOpen, openProject, closeProject, closeAll };
}

export type Launcher = ReturnType<typeof createLauncher>;

interface Request {
    id?: number;
    cmd: string;
    dir?: string;
}

async function dispatch(req: Request, launcher: Launcher): Promise<object> {
    const { id, cmd, dir } = req;
    if (cmd === "recents") return { id, ok: true, recents: launcher.recents() };
    if (cmd === "open") {
        if (!dir) throw new Error("open requires dir");
        return { id, ok: true, ...(await launcher.openProject(dir)) };
    }
    if (cmd === "close") {
        if (!dir) throw new Error("close requires dir");
        await launcher.closeProject(dir);
        return { id, ok: true };
    }
    if (cmd === "closeAll") {
        await launcher.closeAll();
        return { id, ok: true };
    }
    throw new Error(`unknown command: ${cmd}`);
}

/**
 * the control protocol the native shell drives: newline-delimited JSON requests
 * (`{id?, cmd: "recents"|"open"|"close"|"closeAll", dir?}`) in, `{id, ok, ...}` responses out. Streams
 * are injected so the protocol is exercised in-process by tests; the process entry wires stdin/stdout.
 */
export async function runSidecar(
    launcher: Launcher,
    io: { input: AsyncIterable<Uint8Array | string>; write: (line: string) => void },
): Promise<void> {
    const reply = (msg: object) => io.write(JSON.stringify(msg) + "\n");
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of io.input) {
        buf += typeof chunk === "string" ? chunk : decoder.decode(chunk);
        let nl = buf.indexOf("\n");
        while (nl !== -1) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            nl = buf.indexOf("\n");
            if (!line) continue;
            let req: Request;
            try {
                req = JSON.parse(line);
            } catch {
                reply({ ok: false, error: `invalid json: ${line}` });
                continue;
            }
            try {
                reply(await dispatch(req, launcher));
            } catch (e) {
                reply({ id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) });
            }
        }
    }
}

if (import.meta.main) {
    const launcher = createLauncher();
    await runSidecar(launcher, {
        input: process.stdin,
        write: (line) => process.stdout.write(line),
    });
    await launcher.closeAll();
}
