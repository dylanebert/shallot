import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    addRecent,
    createLauncher,
    type Launcher,
    loadRegistry,
    removeRecent,
    runSidecar,
    saveRegistry,
} from "./launcher";

const STUB = resolve(import.meta.dir, "launcher.stub.ts");
const stubCommand = (dir: string, port: number) => ["bun", STUB, dir, String(port)];

const tmpDirs: string[] = [];
const launchers: Launcher[] = [];

function tmp(prefix: string): string {
    const dir = mkdtempSync(join(tmpdir(), prefix));
    tmpDirs.push(dir);
    return dir;
}

function launcher(registry: string): Launcher {
    const l = createLauncher({ registry, command: stubCommand });
    launchers.push(l);
    return l;
}

afterEach(async () => {
    await Promise.all(launchers.map((l) => l.closeAll()));
    launchers.length = 0;
    for (const dir of tmpDirs) rmSync(dir, { recursive: true, force: true });
    tmpDirs.length = 0;
});

test("registry round-trips through disk", () => {
    const file = join(tmp("shallot-reg-"), "projects.json");
    expect(loadRegistry(file).projects).toEqual([]);

    let reg = addRecent({ version: 1, projects: [] }, "/a/one", "2026-06-18T00:00:00.000Z");
    reg = addRecent(reg, "/a/two", "2026-06-18T00:01:00.000Z");
    saveRegistry(reg, file);

    const loaded = loadRegistry(file);
    expect(loaded.projects.map((p) => p.path)).toEqual([resolve("/a/two"), resolve("/a/one")]);
    expect(loaded.projects[0].name).toBe("two");
});

test("addRecent dedupes by path, moves to front, removeRecent drops", () => {
    let reg = addRecent({ version: 1, projects: [] }, "/a/one", "t1");
    reg = addRecent(reg, "/a/two", "t2");
    reg = addRecent(reg, "/a/one", "t3"); // re-open the older project

    expect(reg.projects.map((p) => p.path)).toEqual([resolve("/a/one"), resolve("/a/two")]);
    expect(reg.projects[0].opened).toBe("t3");

    reg = removeRecent(reg, "/a/one");
    expect(reg.projects.map((p) => p.path)).toEqual([resolve("/a/two")]);
});

test("addRecent caps the list at 20, keeping the most recent", () => {
    let reg = { version: 1, projects: [] as never[] } as ReturnType<typeof loadRegistry>;
    for (let i = 0; i < 25; i++) reg = addRecent(reg, `/a/p${i}`, `t${i}`);

    expect(reg.projects).toHaveLength(20);
    expect(reg.projects[0].path).toBe(resolve("/a/p24"));
    expect(reg.projects.some((p) => p.path === resolve("/a/p4"))).toBe(false);
});

test("openProject spawns a server on the right dir, closeProject releases the port", async () => {
    const file = join(tmp("shallot-reg-"), "projects.json");
    const project = tmp("shallot-proj-");
    const l = launcher(file);

    const open = await l.openProject(project);
    expect(open.port).toBeGreaterThan(0);
    expect(open.url).toBe(`http://localhost:${open.port}`);
    expect(l.isOpen(project)).toBe(true);

    // serving the right dir: the stub answers with the dir the launcher spawned it for
    expect(await (await fetch(open.url)).text()).toBe(resolve(project));

    const recents = l.recents();
    expect(recents[0].path).toBe(resolve(project));
    expect(recents[0].exists).toBe(true);

    await l.closeProject(project);
    expect(l.isOpen(project)).toBe(false);
    // port released + child exited: nothing answers on the port anymore
    await expect(fetch(open.url)).rejects.toThrow();
}, 15_000);

test("openProject is idempotent for an already-open project", async () => {
    const file = join(tmp("shallot-reg-"), "projects.json");
    const project = tmp("shallot-proj-");
    const l = launcher(file);

    const a = await l.openProject(project);
    const b = await l.openProject(project);
    expect(b.port).toBe(a.port);
}, 15_000);

test("openProject rejects when the child exits early, surfacing its stderr", async () => {
    const file = join(tmp("shallot-reg-"), "projects.json");
    const l = createLauncher({
        registry: file,
        command: () => ["bun", "-e", "console.error('boom'); process.exit(1)"],
    });
    launchers.push(l);
    await expect(l.openProject(tmp("shallot-proj-"))).rejects.toThrow(/exited early.*boom/s);
}, 15_000);

test("runSidecar dispatches the control protocol over JSON lines", async () => {
    const file = join(tmp("shallot-reg-"), "projects.json");
    const project = tmp("shallot-proj-");
    const l = launcher(file);

    const requests =
        [
            JSON.stringify({ id: 1, cmd: "recents" }),
            JSON.stringify({ id: 2, cmd: "open", dir: project }),
            JSON.stringify({ id: 3, cmd: "close", dir: project }),
            "not json",
            JSON.stringify({ id: 5, cmd: "bogus" }),
        ].join("\n") + "\n";

    const out: string[] = [];
    async function* input() {
        yield requests;
    }
    await runSidecar(l, { input: input(), write: (line) => out.push(line) });

    const res = out.map((line) => JSON.parse(line));
    expect(res[0]).toMatchObject({ id: 1, ok: true });
    expect(Array.isArray(res[0].recents)).toBe(true);
    expect(res[1]).toMatchObject({
        id: 2,
        ok: true,
        url: expect.stringContaining("http://localhost:"),
    });
    expect(res[2]).toMatchObject({ id: 3, ok: true });
    expect(res[3]).toMatchObject({ ok: false });
    expect(res[4]).toMatchObject({ id: 5, ok: false });
}, 15_000);
