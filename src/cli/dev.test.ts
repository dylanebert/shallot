import { expect } from "bun:test";
import { existsSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
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

check(
    "dev passes Vite arguments through",
    {
        claim: "shallot dev preserves Vite flags and values in their original order",
        size: "unit",
        subject: "src/cli/index.ts",
    },
    () => {
        expect(
            parseCliArgs([
                "dev",
                "--port",
                "4012",
                "--strict-port",
                "--no-open",
                "--host",
                "127.0.0.1",
            ]),
        ).toEqual({
            kind: "run",
            subcmd: "dev",
            dir: ".",
            target: undefined,
            release: false,
            portable: false,
            viteArgs: ["--port", "4012", "--strict-port", "--no-open", "--host", "127.0.0.1"],
        });
    },
);

check(
    "the real CLI refuses a project without Vite",
    {
        claim: "shallot build refuses a project with no resolvable Vite and names the devDependency fix without fetching",
        size: "integration",
        subject: ["src/project/vite-command.ts", "src/cli/index.ts"],
        budget: 15_000,
    },
    () => {
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
    },
);

check(
    "Vite commands preserve all remaining arguments",
    {
        claim: "shallot maps dev, build and preview to Vite and preserves each command's arguments",
        size: "unit",
        subject: "src/project/vite-command.ts",
    },
    () => {
        const forwarded = ["--port", "4012", "--mode", "staging"];
        expect(viteCliArgs("dev", forwarded)).toEqual(forwarded);
        expect(viteCliArgs("build", forwarded)).toEqual(["build", ...forwarded]);
        expect(viteCliArgs("preview", forwarded)).toEqual(["preview", ...forwarded]);
    },
);

check(
    "native URL selection prefers Local and falls back to Network",
    {
        claim: "native dev keeps the complete reported Local URL and can use a Network URL alone",
        size: "unit",
        subject: "src/project/vite-command.ts",
    },
    () => {
        const network = "  ➜  Network:   http://192.168.0.139:41989/game/";
        const local = "  ➜  Local:     http://localhost:41989/game/";
        expect(viteUrlFromLines([network])).toBe("http://192.168.0.139:41989/game/");
        expect(viteUrlFromLines([local])).toBe("http://localhost:41989/game/");
        expect(viteUrlFromLines([network, local])).toBe("http://localhost:41989/game/");
    },
);

check(
    "native dev accepts a Network-only Vite URL report",
    {
        claim: "native dev starts when Vite reports only a Network URL",
        size: "integration",
        subject: "src/project/vite-command.ts",
        budget: 3_000,
    },
    async () => {
        const project = mkdtempSync(join(tmpdir(), "shallot-vite-network-url-"));
        const cli = join(project, "vite-fake.mjs");
        writeFileSync(
            cli,
            `console.log("  ➜  Network:   http://192.168.0.139:41989/game/"); setInterval(() => {}, 1000);`,
        );
        let server: Awaited<ReturnType<typeof startViteDev>> | undefined;
        try {
            server = await startViteDev(project, [], cli);
            expect(server.url).toBe("http://192.168.0.139:41989/game/");
        } finally {
            await server?.close();
            rmSync(project, { recursive: true, force: true });
        }
    },
);

check(
    "dev preserves the Vite base path in the native URL",
    {
        claim: "native dev loads the complete local Vite URL when the project uses a base path",
        size: "integration",
        subject: "src/project/vite-command.ts",
        budget: 15_000,
    },
    async () => {
        const root = resolve(import.meta.dir, "../..");
        const project = mkdtempSync(join(tmpdir(), "shallot-vite-base-path-"));
        symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
        writeFileSync(join(project, "package.json"), JSON.stringify({ type: "module" }));
        writeFileSync(join(project, "index.html"), "<!doctype html><title>BASE PATH</title>\n");
        writeFileSync(join(project, "vite.config.ts"), `export default { base: "/game/" };\n`);
        let server: Awaited<ReturnType<typeof startViteDev>> | undefined;
        try {
            server = await startViteDev(project, ["--host", "127.0.0.1", "--port", "0"]);
            expect(new URL(server.url).pathname).toBe("/game/");
            const response = await waitForPage(server.url);
            expect(response.status).toBe(200);
            expect(await response.text()).toContain("BASE PATH");
        } finally {
            await server?.close();
            rmSync(project, { recursive: true, force: true });
        }
    },
);

check(
    "dev delegates to the project's Vite config",
    {
        claim: "shallot dev --port and vite --port serve the same page and project config",
        size: "integration",
        subject: "src/cli/index.ts",
        budget: 20_000,
    },
    async () => {
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
                resolveViteCli(project),
                "--port",
                String(port),
                "--strictPort",
                "--host",
                "127.0.0.1",
            ]);
            const shallot = await serve([
                process.execPath,
                join(root, "bin/shallot.ts"),
                "dev",
                "--port",
                String(port),
                "--strictPort",
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
    },
);

check(
    "dev does not synthesize a missing page",
    {
        claim: "a project without its own index.html is not served a generated Shallot page",
        size: "integration",
        subject: "src/cli/index.ts",
        budget: 15_000,
    },
    async () => {
        const root = resolve(import.meta.dir, "../..");
        const project = mkdtempSync(join(tmpdir(), "shallot-no-generated-page-"));
        symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
        writeFileSync(
            join(project, "shallot.json"),
            JSON.stringify({ kind: "recipe", plugins: {} }),
        );
        const port = await freePort();
        const child = Bun.spawn(
            [
                process.execPath,
                join(root, "bin/shallot.ts"),
                "dev",
                "--port",
                String(port),
                "--strictPort",
                "--host",
                "127.0.0.1",
            ],
            { cwd: project, stdout: "ignore", stderr: "ignore", detached: true },
        );
        try {
            const response = await waitForPage(`http://127.0.0.1:${port}/`);
            expect(response.status).toBe(404);
            expect(await response.text()).not.toContain('<canvas id="canvas"></canvas>');
            expect(existsSync(join(project, "index.html"))).toBe(false);
        } finally {
            try {
                process.kill(-child.pid, "SIGTERM");
            } catch {}
            child.kill();
            await child.exited;
            rmSync(project, { recursive: true, force: true });
        }
    },
);

check(
    "native dev reports the project's Vite URL",
    {
        claim: "native dev receives the actual local URL reported by the project's Vite server",
        size: "integration",
        subject: "src/project/vite-command.ts",
        budget: 15_000,
    },
    async () => {
        const root = resolve(import.meta.dir, "../..");
        const project = mkdtempSync(join(tmpdir(), "shallot-vite-native-url-"));
        symlinkSync(join(root, "node_modules"), join(project, "node_modules"), "dir");
        writeFileSync(join(project, "index.html"), "<!doctype html><title>VITE URL</title>\n");
        let server: Awaited<ReturnType<typeof startViteDev>> | undefined;
        try {
            server = await startViteDev(project, ["--port", "0", "--strictPort"]);
            const response = await waitForPage(server.url);
            expect(server.url).toMatch(/^http:\/\/(localhost|127\.0\.0\.1|\[::1\]):\d+\/?$/);
            expect(await response.text()).toContain("VITE URL");
        } finally {
            await server?.close();
            rmSync(project, { recursive: true, force: true });
        }
    },
);
