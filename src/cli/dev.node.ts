import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    realpathSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import {
    resolveViteCli,
    startViteDev,
    viteCliArgs,
    viteUrlFromLines,
} from "../project/vite-command";
import { parseCliArgs } from "./index";

async function freePort(): Promise<number> {
    const server = createServer();
    await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", () => resolve());
    });
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("no TCP address");
    await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
    );
    return address.port;
}

async function waitForPage(url: string): Promise<Response> {
    const deadline = Date.now() + 10_000;
    let lastError: unknown;
    while (Date.now() < deadline) {
        try {
            return await fetch(url);
        } catch (error) {
            lastError = error;
        }
        await Bun.sleep(50);
    }
    throw new Error(`Vite did not serve ${url}: ${String(lastError)}`);
}

test("shallot dev preserves Vite flags and values in their original order", () => {
    expect(
        parseCliArgs(["dev", "--port", "4012", "--mode", "test", "--host", "127.0.0.1"]),
    ).toEqual({
        kind: "run",
        subcmd: "dev",
        dir: ".",
        target: undefined,
        release: false,
        portable: false,
        viteArgs: ["--port", "4012", "--mode", "test", "--host", "127.0.0.1"],
    });
});

test("shallot build refuses a project with no resolvable Vite and names the devDependency fix without fetching", () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-no-vite-"));
    writeFileSync(
        join(project, "index.html"),
        "<!doctype html><title>No Vite dependency</title>\n",
    );
    try {
        for (let current = project; ; current = dirname(current)) {
            expect(existsSync(join(current, "node_modules"))).toBe(false);
            const parent = dirname(current);
            if (parent === current) break;
        }
        const result = Bun.spawnSync(
            [process.execPath, join(root, "bin/shallot.ts"), "build", project],
            { cwd: root, stdout: "pipe", stderr: "pipe" },
        );
        expect(result.exitCode).toBe(1);
        expect(result.stdout.toString()).toBe("");
        expect(result.stderr.toString()).toContain("Add vite as a devDependency");
        expect(result.stderr.toString()).toContain("bun add -d vite");
        expect(existsSync(join(project, "dist"))).toBe(false);
    } finally {
        rmSync(project, { recursive: true, force: true });
    }
});

test("a bun-linked Shallot consumer without Vite is told to add Vite as a devDependency", () => {
    const root = resolve(import.meta.dir, "../..");
    const consumer = mkdtempSync(join(tmpdir(), "shallot-linked-consumer-"));
    const nodeModules = join(consumer, "node_modules");
    const linkedPackage = join(nodeModules, "@dylanebert", "shallot");
    const linkedBin = join(nodeModules, ".bin", "shallot");
    mkdirSync(join(nodeModules, "@dylanebert"), { recursive: true });
    mkdirSync(join(nodeModules, ".bin"), { recursive: true });
    symlinkSync(root, linkedPackage, "dir");
    symlinkSync("../@dylanebert/shallot/bin/shallot.ts", linkedBin);
    writeFileSync(join(consumer, "index.html"), "<!doctype html><title>Linked app</title>\n");
    try {
        for (let current = consumer; ; current = dirname(current)) {
            expect(existsSync(join(current, "node_modules", "vite", "package.json"))).toBe(false);
            const parent = dirname(current);
            if (parent === current) break;
        }
        const result = Bun.spawnSync([process.execPath, linkedBin, "build"], {
            cwd: consumer,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(result.exitCode).toBe(1);
        expect(result.stderr.toString()).toContain("Add vite as a devDependency");
        expect(result.stderr.toString()).toContain("bun add -d vite");
        expect(existsSync(join(consumer, "dist"))).toBe(false);
    } finally {
        rmSync(consumer, { recursive: true, force: true });
    }
});

test("shallot maps dev, build and preview to Vite and preserves each command's arguments", () => {
    const forwarded = ["--port", "4012", "--mode", "staging"];
    expect(viteCliArgs("dev", forwarded)).toEqual(forwarded);
    expect(viteCliArgs("build", forwarded)).toEqual(["build", ...forwarded]);
    expect(viteCliArgs("preview", forwarded)).toEqual(["preview", ...forwarded]);
});

test("the real Shallot CLI finds Vite hoisted above a workspace app reached through a directory symlink", () => {
    const root = resolve(import.meta.dir, "../..");
    const fixture = mkdtempSync(join(tmpdir(), "shallot-workspace-symlink-"));
    const workspace = join(fixture, "workspace");
    const projectPath = join(workspace, "packages", "game");
    const linkedProject = join(fixture, "consumer", "game");
    mkdirSync(projectPath, { recursive: true });
    const realProject = realpathSync(projectPath);
    mkdirSync(dirname(linkedProject), { recursive: true });
    symlinkSync(join(root, "node_modules"), join(workspace, "node_modules"), "dir");
    symlinkSync(realProject, linkedProject, "dir");
    writeFileSync(join(realProject, "index.html"), "<!doctype html><title>Workspace</title>\n");
    try {
        expect(realpathSync(linkedProject)).toBe(realProject);
        expect(existsSync(join(workspace, "node_modules", "vite", "package.json"))).toBe(true);
        for (let current = linkedProject; ; current = dirname(current)) {
            expect(existsSync(join(current, "node_modules", "vite", "package.json"))).toBe(false);
            const parent = dirname(current);
            if (parent === current) break;
        }
        const result = Bun.spawnSync(
            [process.execPath, join(root, "bin/shallot.ts"), "build", linkedProject],
            { cwd: root, stdout: "pipe", stderr: "pipe" },
        );
        expect(result.exitCode).toBe(0);
        expect(result.stdout.toString()).toContain("vite v");
        expect(existsSync(join(realProject, "dist", "index.html"))).toBe(true);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("shallot dev, build and preview run the Vite declared by the project", async () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-project-vite-"));
    const nodeModules = join(project, "node_modules");
    const packageManifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    const cli = join(root, "bin/shallot.ts");
    symlinkSync(join(root, "node_modules"), nodeModules, "dir");
    writeFileSync(
        join(project, "package.json"),
        JSON.stringify({
            name: "project-vite-contract",
            private: true,
            type: "module",
            devDependencies: { vite: packageManifest.devDependencies.vite },
        }),
    );
    writeFileSync(join(project, "index.html"), "<!doctype html><title>PROJECT VITE</title>\n");
    try {
        expect(resolveViteCli(project)).toBe(
            join(realpathSync(project), "node_modules/vite/bin/vite.js"),
        );

        const build = Bun.spawnSync([process.execPath, cli, "build"], {
            cwd: project,
            stdout: "pipe",
            stderr: "pipe",
        });
        const buildOutput = `${build.stdout.toString()}${build.stderr.toString()}`;
        expect(build.exitCode, buildOutput).toBe(0);
        expect(buildOutput).toContain("vite v");
        expect(existsSync(join(project, "dist/index.html"))).toBe(true);

        const serve = async (command: "dev" | "preview") => {
            const port = await freePort();
            const child = Bun.spawn(
                [process.execPath, cli, command, "--host", "127.0.0.1", "--port", String(port)],
                { cwd: project, stdout: "ignore", stderr: "ignore", detached: true },
            );
            try {
                const response = await waitForPage(`http://127.0.0.1:${port}/`);
                expect(response.status).toBe(200);
                expect(await response.text()).toContain("PROJECT VITE");
            } finally {
                try {
                    process.kill(-child.pid, "SIGTERM");
                } catch {}
                child.kill();
                await child.exited;
            }
        };
        await serve("dev");
        await serve("preview");
    } finally {
        rmSync(project, { recursive: true, force: true });
    }
});

test("native dev keeps the complete reported Local URL and can use a Network URL alone", () => {
    const network = "  ➜  Network:   http://192.168.0.139:41989/game/";
    const local = "  ➜  Local:     http://localhost:41989/game/";
    expect(viteUrlFromLines([network])).toBe("http://192.168.0.139:41989/game/");
    expect(viteUrlFromLines([local])).toBe("http://localhost:41989/game/");
    expect(viteUrlFromLines([network, local])).toBe("http://localhost:41989/game/");
});

test("native dev loads the complete local Vite URL when the project uses a base path", async () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-vite-base-path-"));
    symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
    writeFileSync(join(project, "package.json"), JSON.stringify({ type: "module" }));
    writeFileSync(join(project, "index.html"), "<!doctype html><title>BASE PATH</title>\n");
    writeFileSync(join(project, "vite.config.ts"), `export default { base: "/game/" };\n`);
    let server: Awaited<ReturnType<typeof startViteDev>> | undefined;
    try {
        server = await startViteDev(
            project,
            ["--host", "127.0.0.1", "--port", "0"],
            join(root, "node_modules/vite/bin/vite.js"),
        );
        expect(new URL(server.url).pathname).toBe("/game/");
        const response = await waitForPage(server.url);
        expect(response.status).toBe(200);
        expect(await response.text()).toContain("BASE PATH");
    } finally {
        await server?.close();
        rmSync(project, { recursive: true, force: true });
    }
});

test("shallot dev --port and vite --port serve the same page and project config", async () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-vite-delegation-"));
    symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
    const port = await freePort();
    writeFileSync(join(project, "index.html"), "<!doctype html><title>OWNED PAGE</title>\n");
    writeFileSync(
        join(project, "vite.config.ts"),
        `export default { server: { headers: { "X-Project-Vite-Config": "same" } } };\n`,
    );
    const serve = async (command: string[]) => {
        const child = Bun.spawn(command, {
            cwd: project,
            stdout: "ignore",
            stderr: "ignore",
            detached: true,
        });
        try {
            const response = await waitForPage(`http://127.0.0.1:${port}/`);
            return {
                status: response.status,
                header: response.headers.get("X-Project-Vite-Config"),
                body: await response.text(),
            };
        } finally {
            try {
                process.kill(-child.pid, "SIGTERM");
            } catch {}
            child.kill();
            await child.exited;
        }
    };
    try {
        const vite = await serve([
            process.execPath,
            join(root, "node_modules/vite/bin/vite.js"),
            "--port",
            String(port),
            "--host",
            "127.0.0.1",
        ]);
        const shallot = await serve([
            process.execPath,
            join(root, "bin/shallot.ts"),
            "dev",
            "--port",
            String(port),
            "--host",
            "127.0.0.1",
        ]);
        expect(vite).toEqual(shallot);
        expect(vite.status).toBe(200);
        expect(vite.header).toBe("same");
        expect(vite.body).toContain("OWNED PAGE");
    } finally {
        rmSync(project, { recursive: true, force: true });
    }
});

test("Vite returns no entry page when the project has no index.html", async () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-no-entry-page-"));
    symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
    const port = await freePort();
    const child = Bun.spawn(
        [
            process.execPath,
            join(root, "bin/shallot.ts"),
            "dev",
            "--port",
            String(port),
            "--host",
            "127.0.0.1",
        ],
        { cwd: project, stdout: "ignore", stderr: "ignore", detached: true },
    );
    try {
        const response = await waitForPage(`http://127.0.0.1:${port}/`);
        expect(response.status).toBe(404);
        expect(existsSync(join(project, "index.html"))).toBe(false);
    } finally {
        try {
            process.kill(-child.pid, "SIGTERM");
        } catch {}
        child.kill();
        await child.exited;
        rmSync(project, { recursive: true, force: true });
    }
});

test("native dev receives the actual local URL reported by the project's Vite server", async () => {
    const root = resolve(import.meta.dir, "../..");
    const project = mkdtempSync(join(tmpdir(), "shallot-vite-native-url-"));
    symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
    writeFileSync(join(project, "index.html"), "<!doctype html><title>VITE URL</title>\n");
    let server: Awaited<ReturnType<typeof startViteDev>> | undefined;
    try {
        server = await startViteDev(
            project,
            ["--port", "0"],
            join(root, "node_modules/vite/bin/vite.js"),
        );
        const response = await waitForPage(server.url);
        expect(server.url).toMatch(/^http:\/\/(localhost|127\.0\.0\.1|\[::1\]):\d+\/?$/);
        expect(await response.text()).toContain("VITE URL");
    } finally {
        await server?.close();
        rmSync(project, { recursive: true, force: true });
    }
});
