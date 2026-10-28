import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { spawn } from "node:child_process";
import {
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    symlinkSync,
    unlinkSync,
    writeFileSync,
} from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "../../scripts/bun-child";
import { packedInstall } from "../../scripts/packed-install-spawn";
import { publishTestChromium } from "../../scripts/publish-browser";
import { removeTestDirectory } from "../../scripts/test-directory";
import {
    cloudflareConfig,
    discoverWranglerConfig,
    resolveWrangler,
    WRANGLER_VERSION,
} from "./publish-cloudflare";
import { verifyPublishedGame } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const repo = resolve(import.meta.dir, "../..");

async function freePort(): Promise<number> {
    const server = createServer();
    await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("No local port");
    await new Promise<void>((done, reject) =>
        server.close((error) => (error ? reject(error) : done())),
    );
    return address.port;
}

function cleanEnv(home: string): NodeJS.ProcessEnv {
    return {
        ...Object.fromEntries(
            Object.entries(process.env).filter(([name]) => !/^(?:CLOUDFLARE_|CF_)/.test(name)),
        ),
        HOME: home,
        XDG_CONFIG_HOME: home,
        WRANGLER_SEND_METRICS: "false",
        CI: "true",
    };
}

for (const cause of ["device", "console", "page", "browser", "crash"] as const)
    test(`publish verification fails with the observed ${cause} failure before startup`, async () => {
        const browser = await publishTestChromium();
        const server = createHttpServer((request, response) => {
            if (request.url !== "/") {
                response.writeHead(404).end();
                return;
            }
            response.writeHead(200, {
                "content-type": "text/html",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
            });
            response.end(
                `<link rel="icon" href="data:,"><script>${cause === "device" ? 'throw new Error("WebGPU device creation failed: test device unavailable")' : cause === "console" ? 'console.error("WebGPU device creation failed: test console error")' : 'console.log("ready to terminate")'}</script>`,
            );
        });
        await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
        const address = server.address();
        if (!address || typeof address === "string") throw new Error("No local port");
        browser.on("context", (context) => {
            context.on("page", (page) => {
                page.on("console", (message) => {
                    if (message.text() !== "ready to terminate") return;
                    if (cause === "browser") void browser.close();
                    if (cause === "page") void page.close();
                    if (cause === "crash")
                        void context
                            .newCDPSession(page)
                            .then((session) => session.send("Page.crash"))
                            .catch(() => {});
                });
            });
        });
        try {
            expect(
                await verifyPublishedGame(
                    browser,
                    `http://127.0.0.1:${address.port}`,
                    repo,
                    [],
                ).then(
                    () => undefined,
                    (error: Error) => error.message,
                ),
            ).toMatch(
                cause === "device" || cause === "console"
                    ? "WebGPU device creation failed"
                    : cause === "crash"
                      ? "crashed"
                      : /closed|disconnected/,
            );
        } finally {
            await browser.close();
            await new Promise<void>((done) => server.close(() => done()));
        }
    });

const root = realpathSync.native(mkdtempSync(join(tmpdir(), "shallot-publish-node-")));
const project = join(root, "game");
const home = join(root, "home");
mkdirSync(project);
mkdirSync(home);
const env = cleanEnv(home);
const run = (args: string[], cwd = project) => {
    const result = Bun.spawnSync({ ...bunChild(args), cwd, env, stdout: "pipe", stderr: "pipe" });
    expect(result.exitCode, `${result.stdout}\n${result.stderr}`).toBe(0);
    return result.stdout.toString().trim();
};
let spec: string;
beforeAll(() => {
    const archive = run(["pm", "pack", "--ignore-scripts", "--quiet", "--destination", root], repo);
    spec = `file:../${basename(archive)}`;
    writeFileSync(
        join(project, "package.json"),
        JSON.stringify({
            name: "publish-fixture",
            dependencies: { "@dylanebert/shallot": spec },
            devDependencies: {
                wrangler: WRANGLER_VERSION,
                vite: JSON.parse(readFileSync(join(repo, "package.json"), "utf8")).devDependencies
                    .vite,
                playwright: JSON.parse(readFileSync(join(repo, "package.json"), "utf8"))
                    .peerDependencies.playwright,
            },
        }),
    );
    packedInstall(
        ["install", "--linker=isolated", "--no-progress"],
        project,
        "publish CLI install",
    );
});
afterAll(() => removeTestDirectory(root));

for (const template of ["minimal", "physics"])
    test(`a packed minimal scaffold ${template === "physics" ? "with physics" : "without physics"} starts through pinned Wrangler assets, dry-runs without credentials, and changes only its link`, async () => {
        const cli = join(project, "node_modules/@dylanebert/shallot/bin/shallot.ts");
        const scaffold = join(root, `scaffold-${template}`);
        run([cli, "create", scaffold]);
        if (template === "physics") {
            const html = join(scaffold, "index.html");
            writeFileSync(
                html,
                readFileSync(html, "utf8")
                    .replace("runApp, shallotDark", "runApp, shallotDark, StandardPhysicsPlugin")
                    .replace("plugins: [Welcome]", "plugins: [StandardPhysicsPlugin, Welcome]")
                    .replace(
                        'from "@dylanebert/shallot";',
                        'from "@dylanebert/shallot"; import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";',
                    )
                    .replace(
                        "await runApp({",
                        "const app = await runApp({ setup: world => { world.resource(PhysicsWorldDefinition).workerCount = 2; },",
                    )
                    .replace(
                        "loading: shallotDark({ bootstrap }) });",
                        "loading: shallotDark({ bootstrap }) });\n            app.world.tick();",
                    ),
            );
            expect(readFileSync(html, "utf8")).toContain("app.world.tick();");
        }
        const path = join(scaffold, "package.json");
        const manifest = JSON.parse(readFileSync(path, "utf8"));
        manifest.dependencies["@dylanebert/shallot"] = spec;
        manifest.devDependencies.wrangler = WRANGLER_VERSION;
        writeFileSync(path, JSON.stringify(manifest, null, 4));
        symlinkSync(join(project, "node_modules"), join(scaffold, "node_modules"), "junction");
        const installed = join(scaffold, "node_modules/@dylanebert/shallot");
        const publish = (await import(
            join(installed, "src/cli/publish.ts")
        )) as typeof import("./publish");
        const verify = (await import(
            join(installed, "src/cli/publish-verify.ts")
        )) as typeof import("./publish-verify");
        const build = (await import(
            join(installed, "src/project/build.ts")
        )) as typeof import("../project/build");
        const snapshot = (relative = "", files = new Map<string, Buffer<ArrayBuffer>>()) => {
            for (const entry of readdirSync(join(scaffold, relative), {
                withFileTypes: true,
            })) {
                if (entry.name === "node_modules") continue;
                const file = relative ? `${relative}/${entry.name}` : entry.name;
                if (entry.isDirectory()) snapshot(file, files);
                else files.set(file, readFileSync(join(scaffold, file)));
            }
            return files;
        };
        const before = snapshot();
        const directory = join(root, `web-${template}`);
        await build.buildWeb(scaffold, { outDir: directory, stdout: 2 });
        const assets = publish.publishAssets(directory);
        publish.preflightPublish(assets, { files: 20_000, bytes: 25 * 1024 * 1024 });
        writeFileSync(join(directory, "_headers"), publish.publishHeaders());
        const link = {
            target: "cloudflare" as const,
            name: "shallot-publish-local",
            accountId: "a".repeat(32),
        };
        const config = join(root, "wrangler.json");
        writeFileSync(config, JSON.stringify(cloudflareConfig(directory, link)));
        const wrangler = resolveWrangler(scaffold)!;
        const dry = Bun.spawnSync(["node", wrangler, "deploy", "--config", config, "--dry-run"], {
            cwd: scaffold,
            env,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(dry.exitCode, `${dry.stdout}\n${dry.stderr}`).toBe(0);
        expect(dry.stdout.toString()).toContain("--dry-run: exiting now");
        if (template === "minimal") {
            const ancestor = join(root, "wrangler.json");
            // biome-ignore lint/style/useNamingConvention: Wrangler owns its configuration field names.
            const conflicting = { name: link.name, account_id: "b".repeat(32) };
            writeFileSync(ancestor, JSON.stringify(conflicting));
            expect(discoverWranglerConfig(scaffold)).toBe(ancestor);
            const probe = () => {
                const result = Bun.spawnSync(
                    [
                        "node",
                        "-e",
                        `const {unstable_readConfig}=require(${JSON.stringify(join(scaffold, "node_modules/wrangler/wrangler-dist/cli.js"))}); console.log(unstable_readConfig({}).account_id);`,
                    ],
                    {
                        cwd: scaffold,
                        env: { ...env, CLOUDFLARE_ACCOUNT_ID: link.accountId },
                        stdout: "pipe",
                        stderr: "pipe",
                    },
                );
                expect(result.exitCode, result.stderr.toString()).toBe(0);
                return result.stdout.toString().trim();
            };
            // Plain commands are withheld because Wrangler resolves the ancestor's account.
            expect(probe()).toBe(conflicting.account_id);
            unlinkSync(ancestor);
            expect(discoverWranglerConfig(scaffold)).toBeUndefined();
            for (const args of [
                ["login", "--help"],
                ["login", "--device", "--help"],
                ["dev", "--assets", directory, "--help"],
                ["rollback", "--name", link.name, "--help"],
                ["delete", "--name", link.name, "--dry-run"],
            ]) {
                const parsed = Bun.spawnSync(["node", wrangler, ...args], {
                    cwd: scaffold,
                    env: { ...env, CLOUDFLARE_ACCOUNT_ID: link.accountId },
                    stdout: "pipe",
                    stderr: "pipe",
                });
                expect(
                    parsed.exitCode,
                    `${args.join(" ")}: ${parsed.stdout}\n${parsed.stderr}`,
                ).toBe(0);
                if (args[0] === "delete")
                    expect(parsed.stdout.toString()).toContain("--dry-run: exiting now");
            }
        }
        writeFileSync(config, JSON.stringify(cloudflareConfig(directory, link)));
        const port = await freePort();
        const child = spawn(
            "node",
            [
                wrangler,
                "dev",
                "--config",
                config,
                "--port",
                String(port),
                "--inspector-port",
                "0",
                "--ip",
                "127.0.0.1",
                "--persist-to",
                join(root, "state"),
            ],
            { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] },
        );
        let logs = "";
        child.stdout.on("data", (chunk) => {
            logs += chunk;
        });
        child.stderr.on("data", (chunk) => {
            logs += chunk;
        });
        const exited = new Promise<void>((done) => child.once("exit", () => done()));
        let browser: Awaited<ReturnType<typeof publishTestChromium>> | undefined;
        try {
            browser = await publishTestChromium();
            const newContext = browser.newContext.bind(browser);
            browser.newContext = async (...args) => {
                const context = await newContext(...args);
                await context.addInitScript(() =>
                    Object.defineProperty(navigator, "hardwareConcurrency", { value: 2 }),
                );
                return context;
            };
            const url = `http://127.0.0.1:${port}`;
            const deadline = Date.now() + TIMEOUT.startup;
            while (true) {
                try {
                    if ((await fetch(url)).ok) break;
                } catch {
                    /* not listening yet */
                }
                if (child.exitCode !== null || Date.now() > deadline)
                    throw new Error(`Wrangler did not start:\n${logs}`);
                await Bun.sleep(100);
            }
            const rootResponse = await fetch(url);
            expect(rootResponse.headers.get("cache-control")).toBe("no-cache");
            const hashed = assets.find(
                (asset) => asset.path.startsWith("assets/") && asset.path.endsWith(".js"),
            );
            expect(hashed).toBeDefined();
            expect((await fetch(`${url}/${hashed!.path}`)).headers.get("cache-control")).toContain(
                "immutable",
            );
            const verification = await verify.verifyPublishedGame(browser, url, directory, assets);
            console.info(`publish ${template} local verification:`, verification);
            if (template === "physics") {
                expect(verification.artifact).toBe("shared");
                expect(verification.physicsStepped).toBe(true);
            } else expect(verification.threads).toBeUndefined();
            publish.savePublishLink(scaffold, link);
            for (const [file, bytes] of before) {
                if (file !== "package.json")
                    expect(readFileSync(join(scaffold, file)), file).toEqual(bytes);
            }
            expect([...snapshot().keys()].sort()).toEqual([...before.keys()].sort());
            const after = JSON.parse(readFileSync(path, "utf8"));
            expect(after).toEqual({ ...manifest, shallot: { publish: link } });
            const refusal = Bun.spawnSync({
                ...bunChild([join(installed, "bin/shallot.ts"), "publish"]),
                cwd: scaffold,
                env,
                stdin: "ignore",
                stdout: "pipe",
                stderr: "pipe",
                timeout: TIMEOUT.startup,
            });
            expect(refusal.exitCode).toBe(1);
            expect(refusal.stderr.toString()).toContain("CLOUDFLARE_API_TOKEN");
            expect(refusal.stdout.toString()).toBe("");
        } finally {
            await browser?.close();
            child.kill("SIGTERM");
            await exited;
        }
    });
