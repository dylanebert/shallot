import { spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { createInterface } from "node:readline/promises";
import { detectAgenticEnvironment } from "am-i-vibing";
import ci from "ci-info";
import { buildWeb, CROSS_ORIGIN_ISOLATION } from "../project";
import {
    cloudflareConfig,
    cloudflareDeployment,
    type PublishLink,
    resolveWrangler,
    runWrangler,
    WRANGLER_VERSION,
    workerExists,
} from "./publish-cloudflare";
import { publishProgress } from "./publish-output";
import { type PublishAsset, projectChromium, verifyPublishedGame } from "./publish-verify";

export function readPublishLink(projectDir: string): PublishLink | undefined {
    const manifest = JSON.parse(readFileSync(join(projectDir, "package.json"), "utf8"));
    const link = manifest.shallot?.publish;
    if (link === undefined) return;
    if (
        !link ||
        typeof link !== "object" ||
        link.target !== "cloudflare" ||
        typeof link.name !== "string" ||
        !validWorkerName(link.name) ||
        typeof link.accountId !== "string" ||
        !/^[a-f0-9]{32}$/.test(link.accountId)
    )
        throw new Error(
            "Invalid shallot.publish in package.json: expected target cloudflare, name, and accountId.",
        );
    return link;
}

export function savePublishLink(projectDir: string, link: PublishLink): void {
    const path = join(projectDir, "package.json");
    const manifest = JSON.parse(readFileSync(path, "utf8"));
    manifest.shallot = { ...manifest.shallot, publish: link };
    writeFileSync(path, `${JSON.stringify(manifest, null, 4)}\n`);
}

export function validWorkerName(name: string): boolean {
    return /^[a-z0-9][a-z0-9-]{0,62}$/.test(name);
}

export function publishInteraction(env: NodeJS.ProcessEnv, tty: boolean): boolean {
    return (
        tty &&
        !ci.isCI &&
        (!env.CI || env.CI === "false") &&
        detectAgenticEnvironment({ env, processAncestry: [] }).type !== "agent"
    );
}

export function publishAssets(directory: string): PublishAsset[] {
    const files: PublishAsset[] = [];
    const visit = (relative: string) => {
        for (const entry of readdirSync(join(directory, relative), { withFileTypes: true })) {
            const path = relative ? `${relative}/${entry.name}` : entry.name;
            if (entry.isDirectory()) visit(path);
            else if (entry.isFile())
                files.push({ path, size: statSync(join(directory, path)).size });
            else throw new Error(`${path}: publish refuses symbolic links and special files.`);
        }
    };
    visit("");
    return files.sort((a, b) => a.path.localeCompare(b.path));
}

interface PublishLimits {
    files: number;
    bytes: number;
    reserved?: readonly string[];
}

export function preflightPublish(assets: PublishAsset[], limits: PublishLimits): void {
    for (const [index, asset] of assets.entries()) {
        if (index >= limits.files)
            throw new Error(
                `${asset.path}: exceeds the target's ${limits.files.toLocaleString("en-US")} file limit.`,
            );
        if (asset.size > limits.bytes)
            throw new Error(
                `${asset.path}: ${asset.size} bytes exceeds the target's ${limits.bytes} byte (${limits.bytes / 1024 / 1024} MiB) file limit.`,
            );
    }
    if (!assets.some((asset) => asset.path === "index.html"))
        throw new Error("The web build has no index.html.");
    for (const reserved of limits.reserved ?? []) {
        if (assets.some((asset) => asset.path === reserved))
            throw new Error(
                `${reserved}: reserved by publish; remove it from the project's web output.`,
            );
    }
}

export function publishHeaders(): string {
    return `/*\n${Object.entries(CROSS_ORIGIN_ISOLATION)
        .map(([name, value]) => `  ${name}: ${value}`)
        .join(
            "\n",
        )}\n/\n  Cache-Control: no-cache\n/index.html\n  Cache-Control: no-cache\n/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n`;
}

async function ask(question: string, fallback?: string): Promise<string> {
    const input = createInterface({ input: process.stdin, output: process.stderr });
    try {
        return (
            (await input.question(`${question}${fallback ? ` [${fallback}]` : ""}: `)).trim() ||
            fallback ||
            ""
        );
    } finally {
        input.close();
    }
}

function installDev(projectDir: string, dependency: string): void {
    publishProgress(`Adding ${dependency} to the project's devDependencies.`);
    const result = spawnSync("bun", ["add", "--dev", dependency], {
        cwd: projectDir,
        stdio: ["ignore", 2, 2],
    });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`Could not install ${dependency}.`);
}

interface PublishTarget {
    limits: PublishLimits;
    setup(
        projectDir: string,
        work: string,
        interactive: boolean,
        saved?: PublishLink,
    ): Promise<{
        link: PublishLink;
        deploy(directory: string): Promise<{ url: string; buildUrl: string }>;
        commands: string;
    }>;
    headers(directory: string): void;
}

const targets: Record<"cloudflare", PublishTarget> = {
    cloudflare: {
        limits: {
            files: 20_000,
            bytes: 25 * 1024 * 1024,
            reserved: ["_headers", "_redirects", ".assetsignore"],
        },
        headers(directory) {
            writeFileSync(join(directory, "_headers"), publishHeaders());
        },
        async setup(projectDir, work, interactive, saved) {
            let cli = resolveWrangler(projectDir);
            if (!cli) {
                installDev(projectDir, `wrangler@${WRANGLER_VERSION}`);
                cli = resolveWrangler(projectDir)!;
            }
            const configPath = join(work, "wrangler.json");
            writeFileSync(configPath, "{}");
            let serial = 0;
            const run = (args: string[], terminal = false) =>
                runWrangler(
                    projectDir,
                    cli,
                    [...args, "--config", configPath],
                    join(work, `wrangler-${serial++}.jsonl`),
                    terminal,
                );
            const identity = () => {
                const result = run(["whoami", "--json"]);
                let data: { loggedIn: boolean; accounts?: { id: string; name: string }[] };
                try {
                    data = JSON.parse(result.stdout);
                } catch {
                    throw new Error(
                        `Could not determine Wrangler login.\n${result.stderr || result.stdout}`,
                    );
                }
                if (result.status !== 0 && data.loggedIn !== false)
                    throw new Error(`Wrangler whoami failed.\n${result.stderr}`);
                return data;
            };
            let user = identity();
            if (!user.loggedIn) {
                if (!interactive)
                    throw new Error(
                        "Cloudflare login is missing. Set CLOUDFLARE_API_TOKEN for non-interactive publish; run wrangler login in a terminal (wrangler login --device for a remote session).",
                    );
                publishProgress(
                    "Cloudflare login is required. For a remote session, use wrangler login --device.",
                );
                if (
                    run(
                        [
                            "login",
                            ...(process.env.SSH_CONNECTION || process.env.SSH_TTY
                                ? ["--device"]
                                : []),
                        ],
                        true,
                    ).status !== 0
                )
                    throw new Error("Wrangler login failed.");
                user = identity();
                if (!user.loggedIn)
                    throw new Error("Wrangler did not report a login after login completed.");
            }
            const accounts = user.accounts;
            if (!Array.isArray(accounts) || !accounts.length)
                throw new Error("Wrangler reported no accessible Cloudflare accounts.");
            let accountId =
                process.env.CLOUDFLARE_ACCOUNT_ID ||
                (accounts.length === 1 ? accounts[0]!.id : saved?.accountId);
            if (!accountId) {
                if (accounts.length === 1) {
                    accountId = accounts[0]!.id;
                } else {
                    if (!interactive)
                        throw new Error(
                            "More than one Cloudflare account is accessible. Set CLOUDFLARE_ACCOUNT_ID or shallot.publish.accountId in package.json.",
                        );
                    publishProgress(
                        accounts.map((account) => `${account.name}: ${account.id}`).join("\n"),
                    );
                    accountId = await ask("Cloudflare account id");
                }
            }
            if (saved && saved.accountId !== accountId)
                throw new Error(
                    `Resolved Cloudflare account ${accountId} differs from shallot.publish.accountId ${saved.accountId}; refusing to switch accounts.`,
                );
            if (!accounts.some((account) => account.id === accountId))
                throw new Error(`Cloudflare account ${accountId} is not accessible to this login.`);
            publishProgress(
                `Using Cloudflare account ${accounts.find((account) => account.id === accountId)!.name} (${accountId}).`,
            );
            const manifest = JSON.parse(readFileSync(join(projectDir, "package.json"), "utf8"));
            const defaultName = String(manifest.name ?? basename(projectDir))
                .replace(/^@[^/]+\//, "")
                .toLowerCase()
                .replace(/[^a-z0-9-]/g, "-")
                .slice(0, 63);
            let name =
                saved?.name ?? (interactive ? await ask("Worker name", defaultName) : defaultName);
            if (!saved && !interactive)
                publishProgress(`Using project name ${name} as the Worker name.`);
            if (!validWorkerName(name))
                throw new Error(
                    `Invalid Worker name ${JSON.stringify(name)}; use 1–63 lowercase letters, digits and hyphens, starting with a letter or digit.`,
                );
            const link: PublishLink = { target: "cloudflare", name, accountId };
            return {
                link,
                async deploy(directory) {
                    while (true) {
                        if (!validWorkerName(name))
                            throw new Error(
                                `Invalid Worker name ${JSON.stringify(name)}; use 1–63 lowercase letters, digits and hyphens, starting with a letter or digit.`,
                            );
                        writeFileSync(
                            configPath,
                            JSON.stringify(
                                cloudflareConfig(directory, {
                                    target: "cloudflare",
                                    name,
                                    accountId,
                                }),
                            ),
                        );
                        if (saved || !workerExists(run(["versions", "list", "--json"]))) break;
                        const message = `Worker "${name}" already exists in account ${accountId}. No link was saved and nothing was deployed. package.json's shallot.publish.name names the Worker; writing the complete shallot.publish link by hand deliberately authorizes updating it.`;
                        if (!interactive) throw new Error(message);
                        publishProgress(message);
                        name = await ask("Choose another Worker name");
                    }
                    link.name = name;
                    // Wrangler gives no lock: another Worker can appear between the absence check and deploy.
                    const result = run(["deploy"], interactive);
                    if (!interactive) publishProgress(result.stdout, result.stderr);
                    return cloudflareDeployment(result);
                },
                get commands() {
                    return `Preview: bunx wrangler dev --assets <web-build-directory>\nRollback: bunx wrangler rollback --name ${name}\nUnpublish: bunx wrangler delete --name ${name}\nAccount: ${accountId}`;
                },
            };
        },
    },
};

/** Publish a web build; only the provider's CLI reads or writes its credentials. */
export async function publishProject(projectDir: string, targetName?: string): Promise<void> {
    if (targetName !== undefined && targetName !== "cloudflare")
        throw new Error(`Unknown publish target: ${targetName}. Available: cloudflare.`);
    const saved = readPublishLink(projectDir);
    const target = targets[targetName ?? saved?.target ?? "cloudflare"];
    if (!targetName && !saved) publishProgress("Using publish target cloudflare.");
    const work = mkdtempSync(join(tmpdir(), "shallot-publish-"));
    try {
        const directory = join(work, "web");
        const setup = await target.setup(
            projectDir,
            work,
            publishInteraction(process.env, !!process.stdin.isTTY && !!process.stdout.isTTY),
            saved,
        );
        await buildWeb(projectDir, { outDir: directory, stdout: 2 });
        const assets = publishAssets(directory);
        preflightPublish(assets, target.limits);
        target.headers(directory);
        const deployed = await setup.deploy(directory);
        savePublishLink(projectDir, setup.link);
        console.log(deployed.url);
        publishProgress(`Build: ${deployed.buildUrl}\n${setup.commands}`);
        const browser = await projectChromium(projectDir);
        try {
            const verification = await verifyPublishedGame(
                browser,
                deployed.url,
                directory,
                assets,
            );
            publishProgress(
                `${verification.hardware ? "Engine started on a hardware adapter." : "Engine started; loading, headers and assets verified, not hardware play."}${verification.threads === undefined ? "" : ` Physics ${verification.physicsStepped ? "stepped" : "initialized"} on ${verification.threads} threads.`}\nAdapter: ${verification.adapter}`,
            );
        } finally {
            await browser.close();
        }
    } finally {
        rmSync(work, { recursive: true, force: true });
    }
}
