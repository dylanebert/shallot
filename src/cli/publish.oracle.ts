import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { bunChild } from "../../scripts/bun-child";
import { packedInstall } from "../../scripts/packed-install-spawn";
import { browserSessionPath } from "../../scripts/publish-browser-session";
import { cleanupOracleWorkers } from "../../scripts/publish-oracle-cleanup";
import { normalizePath } from "../project";
import { cloudflareConfig, type WranglerResult, workerExists } from "./publish-cloudflare";
import { BUTLER_VERSION } from "./publish-itch";

async function confirmPlay(url: string | null, instruction: string) {
    console.info(instruction);
    if (url) console.info(url);
    const input = createInterface({ input: process.stdin, output: process.stdout });
    try {
        const answer = await input.question("Does it play? [y/N] ");
        expect(
            ["y", "yes"].includes(answer.trim().toLowerCase()),
            `Play judgment failed: the person answered ${JSON.stringify(answer)}.`,
        ).toBe(true);
    } finally {
        input.close();
    }
}

// Real Cloudflare boundary: select with -t 'personal:.*Cloudflare' or -t 'CI:.*Cloudflare'.
// Successful trials remove scratch logins and delete their Workers; failures retain them for inspection.
for (const trial of ["personal", "CI"] as const) {
    test(
        `${trial}: a packed scaffold publishes to Cloudflare`,
        async () => {
            const personal = trial === "personal";
            if (personal && (!process.stdin.isTTY || !process.stdout.isTTY))
                throw new Error(
                    "Run the personal trial from the person's terminal to judge the inline login flow.",
                );
            if (!personal) {
                if (process.stdin.isTTY || process.stdout.isTTY)
                    throw new Error("Run the CI trial without a TTY.");
                for (const name of ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"])
                    if (!process.env[name]) throw new Error(`Set ${name} for the CI trial.`);
            }
            const root = mkdtempSync(join(tmpdir(), "shallot-publish-oracle-"));
            const repo = resolve(import.meta.dir, "../..");
            const installer = join(root, "installer");
            const project = join(root, "game");
            const home = join(root, "home");
            for (const directory of [installer, home]) mkdirSync(directory);
            let qualified = false;
            try {
                console.info(`Oracle project and clean login: ${root}`);
                const trialEnv: NodeJS.ProcessEnv = {
                    ...Object.fromEntries(
                        Object.entries(process.env).filter(
                            ([name]) => !/^(?:CLOUDFLARE_|CF_)/.test(name),
                        ),
                    ),
                    HOME: home,
                    PATH: browserSessionPath(root),
                    XDG_CONFIG_HOME: home,
                    WRANGLER_SEND_METRICS: "false",
                    ...(personal
                        ? { CI: "" }
                        : {
                              CI: "true",
                              CLOUDFLARE_API_TOKEN: process.env.CLOUDFLARE_API_TOKEN,
                              CLOUDFLARE_ACCOUNT_ID: process.env.CLOUDFLARE_ACCOUNT_ID,
                          }),
                };
                const run = (args: string[], cwd: string) => {
                    const result = Bun.spawnSync({
                        ...bunChild(args),
                        cwd,
                        env: trialEnv,
                        stdin: "ignore",
                        stdout: "pipe",
                        stderr: "pipe",
                    });
                    expect(result.exitCode, result.stderr.toString()).toBe(0);
                    process.stderr.write(result.stderr);
                    return result.stdout.toString().trim();
                };
                const archive = run(
                    ["pm", "pack", "--ignore-scripts", "--quiet", "--destination", root],
                    repo,
                );
                const spec = `file:../${basename(archive)}`;
                writeFileSync(
                    join(installer, "package.json"),
                    JSON.stringify({
                        name: "publish-oracle-installer",
                        dependencies: { "@dylanebert/shallot": spec },
                    }),
                );
                packedInstall(
                    ["install", "--linker=isolated", "--no-progress"],
                    installer,
                    "publish oracle CLI install",
                );
                const cli = join(installer, "node_modules/@dylanebert/shallot/bin/shallot.ts");
                run([cli, "create", project], installer);
                const path = join(project, "package.json");
                const manifest = JSON.parse(readFileSync(path, "utf8"));
                manifest.name = `shallot-oracle-${Date.now().toString(36)}`;
                manifest.dependencies["@dylanebert/shallot"] = spec;
                writeFileSync(path, JSON.stringify(manifest, null, 4));
                packedInstall(
                    ["install", "--linker=isolated", "--no-progress"],
                    project,
                    "publish oracle scaffold install",
                );
                const html = join(project, "index.html");
                const marker = `<!-- publish oracle ${crypto.randomUUID()} -->`;
                writeFileSync(html, `${readFileSync(html, "utf8")}\n${marker}\n`);
                const projectCli = join(project, "node_modules/@dylanebert/shallot/bin/shallot.ts");
                const workers = new Set<string>();
                let passed = false;
                try {
                    if (!personal) {
                        const url = run([projectCli, "publish"], project);
                        expect(new URL(url).hostname).toEndWith(".workers.dev");
                        const link = JSON.parse(readFileSync(path, "utf8")).shallot.publish;
                        expect(link.target).toBe("cloudflare");
                        expect(link.accountId).toBe(process.env.CLOUDFLARE_ACCOUNT_ID);
                        expect(new URL(url).hostname).toStartWith(`${link.name}.`);
                        expect(await (await fetch(url)).text()).toContain(marker);
                        console.info(`CI verified link: ${url}\nRetained oracle scratch: ${root}`);
                        passed = true;
                        qualified = true;
                        return;
                    }
                    const first = Bun.spawnSync({
                        ...bunChild([projectCli, "publish"]),
                        cwd: project,
                        env: trialEnv,
                        stdin: "inherit",
                        stdout: "inherit",
                        stderr: "inherit",
                    });
                    expect(first.exitCode).toBe(0);
                    await confirmPlay(
                        null,
                        "Open the play link printed above in a fresh anonymous browser session.",
                    );
                    const linked = JSON.parse(readFileSync(path, "utf8"));
                    const link = linked.shallot.publish;
                    expect(link.target).toBe("cloudflare");
                    expect(linked.devDependencies.wrangler).toBe("4.151.3");
                    const wrangler = join(project, "node_modules/wrangler/bin/wrangler.js");
                    const whoami = spawnSync("node", [wrangler, "whoami", "--json"], {
                        cwd: project,
                        env: trialEnv,
                        encoding: "utf8",
                        stdio: ["ignore", "pipe", "pipe"],
                    });
                    expect(whoami.status, whoami.stderr).toBe(0);
                    const identity = JSON.parse(whoami.stdout);
                    const accountId = link.accountId;
                    expect(accountId).toBeString();
                    expect(
                        identity.accounts.map((account: { id: string }) => account.id),
                    ).toContain(accountId);
                    const config = join(root, "wrangler.json");
                    writeFileSync(
                        config,
                        JSON.stringify(
                            cloudflareConfig(join(root, "unused-assets"), { ...link, accountId }),
                        ),
                    );
                    let serial = 0;
                    const provider = (args: string[]): WranglerResult => {
                        const output = join(root, `provider-${serial++}.jsonl`);
                        const result = spawnSync("node", [wrangler, ...args, "--config", config], {
                            cwd: root,
                            env: { ...trialEnv, WRANGLER_OUTPUT_FILE_PATH: output },
                            encoding: "utf8",
                            stdio: ["ignore", "pipe", "pipe"],
                        });
                        if (result.error) throw result.error;
                        return {
                            status: result.status,
                            stdout: result.stdout ?? "",
                            stderr: result.stderr ?? "",
                            entries: readFileSync(output, "utf8")
                                .trim()
                                .split("\n")
                                .map((line) => JSON.parse(line)),
                        };
                    };
                    const exists = provider(["versions", "list", "--json"]);
                    expect(workerExists(exists)).toBe(true);
                    const absent = provider([
                        "versions",
                        "list",
                        "--json",
                        "--name",
                        `shallot-absent-${crypto.randomUUID().slice(0, 8)}`,
                    ]);
                    expect(workerExists(absent)).toBe(false);
                    console.info("Recorded existence answers:", {
                        existingStatus: exists.status,
                        absentStatus: absent.status,
                        absentFailure: absent.entries.findLast(
                            (entry) => entry.type === "command-failed",
                        )?.code,
                    });
                    const deployments = () => {
                        const result = provider(["deployments", "list", "--json"]);
                        expect(result.status, result.stderr).toBe(0);
                        return JSON.parse(result.stdout) as { id: string }[];
                    };
                    const firstDeployment = deployments().at(-1)!.id;
                    const collision = join(root, "collision");
                    cpSync(project, collision, {
                        recursive: true,
                        filter: (source) =>
                            !normalizePath(source).split("/").includes("node_modules"),
                    });
                    const collisionPath = join(collision, "package.json");
                    const collisionManifest = JSON.parse(readFileSync(collisionPath, "utf8"));
                    collisionManifest.name = link.name;
                    delete collisionManifest.shallot.publish;
                    writeFileSync(collisionPath, JSON.stringify(collisionManifest, null, 4));
                    packedInstall(
                        ["install", "--linker=isolated", "--no-progress"],
                        collision,
                        "publish collision fixture install",
                    );
                    workers.add(collisionManifest.name);
                    const refused = Bun.spawnSync({
                        ...bunChild([
                            join(collision, "node_modules/@dylanebert/shallot/bin/shallot.ts"),
                            "publish",
                        ]),
                        cwd: collision,
                        env: trialEnv,
                        stdin: "ignore",
                        stdout: "pipe",
                        stderr: "pipe",
                    });
                    expect(refused.exitCode).toBe(1);
                    expect(refused.stderr.toString()).toContain(
                        `Worker "${link.name}" already exists`,
                    );
                    expect(refused.stderr.toString()).toContain("shallot.publish.name");
                    expect(
                        JSON.parse(readFileSync(collisionPath, "utf8")).shallot.publish,
                    ).toBeUndefined();
                    expect(deployments().at(-1)!.id).toBe(firstDeployment);
                    const before = readFileSync(html, "utf8");
                    const after = before.replace(
                        /<title>[^<]*<\/title>/,
                        "<title>republished oracle</title>",
                    );
                    expect(after, "Republish must change the scaffold title").not.toBe(before);
                    writeFileSync(html, after);
                    const url = run([projectCli, "publish"], project);
                    expect(new URL(url).hostname).toEndWith(".workers.dev");
                    expect(new URL(url).hostname).toStartWith(`${link.name}.`);
                    expect(deployments().at(-1)!.id).not.toBe(firstDeployment);
                    const served = await (await fetch(url)).text();
                    expect(served).toContain(marker);
                    expect(served).toContain("<title>republished oracle</title>");
                    expect(JSON.parse(readFileSync(path, "utf8")).shallot.publish).toEqual(link);
                    console.info(`Qualified stable link: ${url}\nRetained oracle scratch: ${root}`);
                    passed = true;
                    qualified = true;
                } finally {
                    const linked = JSON.parse(readFileSync(path, "utf8")).shallot?.publish;
                    workers.add(linked?.name ?? manifest.name);
                    const cleanupWrangler = (args: string[], output?: string) => {
                        const result = spawnSync(
                            "node",
                            [join(project, "node_modules/wrangler/bin/wrangler.js"), ...args],
                            {
                                cwd: project,
                                env: {
                                    ...trialEnv,
                                    CI: "true",
                                    ...(linked?.accountId
                                        ? { CLOUDFLARE_ACCOUNT_ID: linked.accountId }
                                        : {}),
                                    ...(output ? { WRANGLER_OUTPUT_FILE_PATH: output } : {}),
                                },
                                encoding: "utf8",
                                stdio: ["ignore", "pipe", "pipe"],
                            },
                        );
                        if (result.error) throw result.error;
                        return result;
                    };
                    cleanupOracleWorkers(
                        workers,
                        passed,
                        linked?.accountId ?? trialEnv.CLOUDFLARE_ACCOUNT_ID,
                        (name) => cleanupWrangler(["delete", "--name", name]),
                        (name) => {
                            const output = join(root, `cleanup-${crypto.randomUUID()}.jsonl`);
                            const result = cleanupWrangler(
                                ["versions", "list", "--json", "--name", name],
                                output,
                            );
                            return workerExists({
                                status: result.status,
                                stdout: result.stdout,
                                stderr: result.stderr,
                                entries: readFileSync(output, "utf8")
                                    .trim()
                                    .split("\n")
                                    .map((line) => JSON.parse(line)),
                            });
                        },
                        console.info,
                    );
                }
            } catch (error) {
                qualified = false;
                throw error;
            } finally {
                if (qualified) rmSync(home, { recursive: true, force: true });
                else
                    console.info(
                        `Retained scratch HOME ${home} may hold a live provider login: revoke it or delete this directory.`,
                    );
            }
        },
        Number.POSITIVE_INFINITY,
    );
}

// Run personal before CI: CI reuses the final personal build without opening a page or prompting.
// Successful trials remove scratch logins; the Restricted page's password is never read here.
for (const trial of ["personal", "CI"] as const) {
    test(
        `${trial}: a packed scaffold publishes to itch.io`,
        async () => {
            const personal = trial === "personal";
            if (personal && (!process.stdin.isTTY || !process.stdout.isTTY))
                throw new Error(
                    "Run the itch.io personal trial from the person's terminal for butler login and play judgments.",
                );
            if (!personal) {
                if (process.stdin.isTTY || process.stdout.isTTY)
                    throw new Error("Run the itch.io CI trial without a TTY.");
                if (!process.env.BUTLER_API_KEY)
                    throw new Error(
                        "Set BUTLER_API_KEY for the itch.io CI trial, after the personal trial passes.",
                    );
            }
            const root = mkdtempSync(join(tmpdir(), "shallot-itch-oracle-"));
            const home = join(root, "home");
            const installer = join(root, "installer");
            const project = join(root, "game");
            for (const directory of [home, installer]) mkdirSync(directory);
            let qualified = false;
            try {
                console.info(`itch.io oracle project and clean login: ${root}`);
                const env: NodeJS.ProcessEnv = {
                    ...Object.fromEntries(
                        Object.entries(process.env).filter(
                            ([name]) => !/^(?:BUTLER_|ITCH_|CLOUDFLARE_|CF_|XDG_)/.test(name),
                        ),
                    ),
                    HOME: home,
                    PATH: browserSessionPath(root),
                    XDG_CONFIG_HOME: join(home, "config"),
                    XDG_CACHE_HOME: join(home, "cache"),
                    XDG_DATA_HOME: join(home, "data"),
                    CI: personal ? "" : "true",
                    ...(personal ? {} : { BUTLER_API_KEY: process.env.BUTLER_API_KEY }),
                };
                const run = (args: string[], cwd: string) => {
                    const result = Bun.spawnSync({
                        ...bunChild(args),
                        cwd,
                        env,
                        stdin: "ignore",
                        stdout: "pipe",
                        stderr: "pipe",
                    });
                    expect(result.exitCode, result.stderr.toString()).toBe(0);
                    process.stderr.write(result.stderr);
                    return result.stdout.toString().trim();
                };
                const archive = run(
                    ["pm", "pack", "--ignore-scripts", "--quiet", "--destination", root],
                    resolve(import.meta.dir, "../.."),
                );
                const spec = `file:../${basename(archive)}`;
                writeFileSync(
                    join(installer, "package.json"),
                    JSON.stringify({
                        name: "itch-oracle-installer",
                        dependencies: { "@dylanebert/shallot": spec },
                    }),
                );
                packedInstall(
                    ["install", "--linker=isolated", "--no-progress"],
                    installer,
                    "itch oracle CLI install",
                );
                run(
                    [
                        join(installer, "node_modules/@dylanebert/shallot/bin/shallot.ts"),
                        "create",
                        project,
                    ],
                    installer,
                );
                const path = join(project, "package.json");
                const manifest = JSON.parse(readFileSync(path, "utf8"));
                const link = { target: "itch", name: "individualkex/shallot-publish-test" };
                manifest.shallot = { ...manifest.shallot, publish: link };
                manifest.dependencies["@dylanebert/shallot"] = spec;
                writeFileSync(path, JSON.stringify(manifest, null, 4));
                packedInstall(
                    ["install", "--linker=isolated", "--no-progress"],
                    project,
                    "itch oracle scaffold install",
                );
                const cli = join(project, "node_modules/@dylanebert/shallot/bin/shallot.ts");
                const url = "https://individualkex.itch.io/shallot-publish-test";
                const html = join(project, "index.html");
                const marker = `<!-- publish oracle ${crypto.randomUUID()} -->`;
                if (personal) writeFileSync(html, `${readFileSync(html, "utf8")}\n${marker}\n`);
                const change = () => {
                    const before = readFileSync(html, "utf8");
                    const after = before.replace(
                        /<title>[^<]*<\/title>/,
                        "<title>republished oracle</title>",
                    );
                    expect(after, "Republish must change the scaffold title").not.toBe(before);
                    writeFileSync(html, after);
                };
                const publish = (terminal: boolean) => {
                    const result = Bun.spawnSync({
                        ...bunChild([cli, "publish"]),
                        cwd: project,
                        env,
                        stdin: terminal ? "inherit" : "ignore",
                        stdout: terminal ? "inherit" : "pipe",
                        stderr: "pipe",
                    });
                    const report = result.stderr.toString();
                    process.stderr.write(report);
                    expect(JSON.parse(readFileSync(path, "utf8")).shallot.publish).toEqual(link);
                    if (!terminal) expect(result.stdout?.toString().trim()).toBe(url);
                    return { code: result.exitCode, report };
                };
                const play = () =>
                    confirmPlay(
                        url,
                        "Open this Restricted page in your own logged-in owner browser. Publish's verifier loads the build's frame document without a login.",
                    );
                const verified = async () => {
                    const result = publish(false);
                    expect(result.code, result.report).toBe(0);
                    expect(result.report).toContain("Engine started");
                    const frame = result.report.match(
                        /Build: (https:\/\/html\.itch\.zone\/\S+)/,
                    )?.[1];
                    expect(frame).toBeDefined();
                    return await (await fetch(`${frame}?v=${crypto.randomUUID()}`)).text();
                };
                if (personal) {
                    const first = publish(true);
                    expect(first.report).toMatch(
                        /Pushed build \d+; waiting for itch\.io to process it\./,
                    );
                    expect(first.report).not.toContain("itch.io already has this build");
                    expect(first.code, first.report).toBe(1);
                    expect(first.report).toContain("itch.io login is required.");
                    expect(first.report).toContain(
                        "Not verified yet: itch.io serves this build once its page is viewed.",
                    );
                    const butler = join(
                        env.XDG_CACHE_HOME!,
                        "shallot",
                        "butler",
                        BUTLER_VERSION,
                        `${process.platform === "linux" ? "linux" : "darwin"}-${process.arch === "x64" ? "amd64" : process.arch}`,
                        "butler",
                    );
                    const version = Bun.spawnSync([butler, "version"], {
                        env,
                        stdout: "pipe",
                        stderr: "pipe",
                    });
                    expect(version.exitCode).toBe(0);
                    expect(`${version.stdout}${version.stderr}`).toContain(`v${BUTLER_VERSION},`);
                    await play();
                    expect(await verified()).toContain(marker);
                    change();
                    const changed = publish(false);
                    expect(changed.report).toMatch(
                        /Pushed build \d+; waiting for itch\.io to process it\./,
                    );
                    expect(changed.report).not.toContain("itch.io already has this build");
                    expect(changed.code, changed.report).toBe(1);
                    expect(changed.report).toContain(
                        "Not verified yet: itch.io serves this build once its page is viewed.",
                    );
                    await play();
                } else change();
                const served = await verified();
                if (personal) expect(served).toContain(marker);
                expect(served).toContain("<title>republished oracle</title>");
                console.info(
                    `Qualified itch.io ${trial} link: ${url}\nRetained oracle scratch: ${root}`,
                );
                qualified = true;
            } finally {
                if (qualified) rmSync(home, { recursive: true, force: true });
                else
                    console.info(
                        `Retained scratch HOME ${home} may hold a live provider login: revoke it or delete this directory.`,
                    );
            }
        },
        Number.POSITIVE_INFINITY,
    );
}
