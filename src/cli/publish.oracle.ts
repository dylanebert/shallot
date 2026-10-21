import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve, sep } from "node:path";
import { createInterface } from "node:readline/promises";
import { bunChild } from "../../scripts/bun-child";
import { packedInstall } from "../../scripts/packed-install-spawn";
import { cloudflareConfig, type WranglerResult, workerExists } from "./publish-cloudflare";

// Manual, real Cloudflare boundary. Run by path from the person's terminal, with
// CLOUDFLARE_API_TOKEN available for the separate CI trial. OAuth starts in a clean HOME.
// The scratch project and login are retained for inspection; no Worker is deleted automatically.
test(
    "a packed scaffold publishes through personal login, refuses collisions, republishes and runs with CI credentials",
    async () => {
        if (!process.stdin.isTTY || !process.stdout.isTTY)
            throw new Error(
                "Run this oracle from the person's terminal to judge the inline login flow.",
            );
        if (!process.env.CLOUDFLARE_API_TOKEN)
            throw new Error(
                "Set CLOUDFLARE_API_TOKEN for the oracle's separate CI-shaped trial; it is removed from the personal-login trial.",
            );
        const root = mkdtempSync(join(tmpdir(), "shallot-publish-oracle-"));
        const repo = resolve(import.meta.dir, "../..");
        const installer = join(root, "installer");
        const project = join(root, "game");
        const home = join(root, "home");
        const ciHome = join(root, "ci-home");
        for (const directory of [installer, home, ciHome]) mkdirSync(directory);
        console.info(`Oracle project and clean login: ${root}`);
        const personalEnv: NodeJS.ProcessEnv = {
            ...Object.fromEntries(
                Object.entries(process.env).filter(([name]) => !/^(?:CLOUDFLARE_|CF_)/.test(name)),
            ),
            HOME: home,
            XDG_CONFIG_HOME: home,
            WRANGLER_SEND_METRICS: "false",
        };
        const run = (args: string[], cwd: string, env = personalEnv) => {
            const result = Bun.spawnSync({
                ...bunChild(args),
                cwd,
                env,
                stdin: "ignore",
                stdout: "pipe",
                stderr: "pipe",
            });
            expect(result.exitCode, result.stderr.toString()).toBe(0);
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
        const projectCli = join(project, "node_modules/@dylanebert/shallot/bin/shallot.ts");
        const first = Bun.spawnSync({
            ...bunChild([projectCli, "publish"]),
            cwd: project,
            env: personalEnv,
            stdin: "inherit",
            stdout: "inherit",
            stderr: "inherit",
        });
        expect(first.exitCode).toBe(0);
        const input = createInterface({ input: process.stdin, output: process.stdout });
        let url: string;
        try {
            url = (
                await input.question(
                    "Open the printed play link anonymously, judge that it plays, then paste that link here: ",
                )
            )
                .trim()
                .replace(/\/$/, "");
        } finally {
            input.close();
        }
        expect(new URL(url).hostname).toEndWith(".workers.dev");
        const linked = JSON.parse(readFileSync(path, "utf8"));
        const link = linked.shallot.publish;
        expect(link.target).toBe("cloudflare");
        expect(linked.devDependencies.wrangler).toBe("4.151.3");
        const wrangler = join(project, "node_modules/wrangler/bin/wrangler.js");
        const whoami = spawnSync("node", [wrangler, "whoami", "--json"], {
            cwd: project,
            env: personalEnv,
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });
        expect(whoami.status, whoami.stderr).toBe(0);
        const identity = JSON.parse(whoami.stdout);
        const accountId =
            link.accountId ??
            (identity.accounts.length === 1 ? identity.accounts[0].id : undefined);
        expect(accountId).toBeString();
        const config = join(root, "wrangler.json");
        writeFileSync(
            config,
            JSON.stringify(cloudflareConfig(join(root, "unused-assets"), { ...link, accountId })),
        );
        let serial = 0;
        const provider = (args: string[]): WranglerResult => {
            const output = join(root, `provider-${serial++}.jsonl`);
            const result = spawnSync("node", [wrangler, ...args, "--config", config], {
                cwd: root,
                env: { ...personalEnv, WRANGLER_OUTPUT_FILE_PATH: output },
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
            absentFailure: absent.entries.findLast((entry) => entry.type === "command-failed")
                ?.code,
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
            filter: (source) => !source.split(sep).includes("node_modules"),
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
        const refused = Bun.spawnSync({
            ...bunChild([
                join(collision, "node_modules/@dylanebert/shallot/bin/shallot.ts"),
                "publish",
            ]),
            cwd: collision,
            env: personalEnv,
            stdin: "ignore",
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(refused.exitCode).toBe(1);
        expect(refused.stderr.toString()).toContain(`Worker "${link.name}" already exists`);
        expect(refused.stderr.toString()).toContain("shallot.publish.name");
        expect(JSON.parse(readFileSync(collisionPath, "utf8")).shallot.publish).toBeUndefined();
        expect(deployments().at(-1)!.id).toBe(firstDeployment);
        const html = join(project, "index.html");
        writeFileSync(
            html,
            readFileSync(html, "utf8").replace(
                "<title>minimal</title>",
                "<title>republished oracle</title>",
            ),
        );
        expect(run([projectCli, "publish"], project)).toBe(url);
        expect(deployments().at(-1)!.id).not.toBe(firstDeployment);
        expect(await (await fetch(url)).text()).toContain("republished oracle");
        const ciEnv = {
            ...process.env,
            CI: "true",
            HOME: ciHome,
            XDG_CONFIG_HOME: ciHome,
            CLOUDFLARE_ACCOUNT_ID: accountId,
        };
        expect(run([projectCli, "publish"], project, ciEnv)).toBe(url);
        expect(JSON.parse(readFileSync(path, "utf8")).shallot.publish).toEqual(link);
        console.info(
            `Qualified stable link: ${url}\nUnpublish deliberately: CLOUDFLARE_ACCOUNT_ID=${accountId} bunx wrangler delete --name ${link.name}\nRetained oracle scratch: ${root}`,
        );
    },
    Number.POSITIVE_INFINITY,
);
