import { spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { createInterface } from "node:readline/promises";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { detectAgenticEnvironment } from "am-i-vibing";
import ci from "ci-info";
import type { Browser } from "playwright";
import { buildWeb, CROSS_ORIGIN_ISOLATION } from "../project";
import {
    CLOUDFLARE_LOGIN_ARGS,
    CLOUDFLARE_LOGIN_DISCLOSURE,
    type CloudflareLink,
    cloudflareConfig,
    cloudflareDeployment,
    discoverWranglerConfig,
    resolveWrangler,
    runWrangler,
    WRANGLER_VERSION,
    workerExists,
} from "./publish-cloudflare";
import {
    BUTLER_LOGIN_REQUIRED,
    BUTLER_STATUS_TIMEOUT,
    ensureButler,
    hasButlerCredentials,
    ITCH_CHANNEL,
    ITCH_HOST,
    type ItchLink,
    itchBuildUrl,
    itchChannel,
    itchPageSettings,
    itchPageUrl,
    itchPush,
    liveItchBuild,
    runButler,
    validItchName,
    waitForItchBuild,
} from "./publish-itch";
import { publishProgress } from "./publish-output";
import {
    type PublishAsset,
    type PublishVerification,
    projectChromium,
    verifyPublishedGame,
} from "./publish-verify";

export type PublishLink = CloudflareLink | ItchLink;

export function readPublishLink(projectDir: string): PublishLink | undefined {
    const manifest = JSON.parse(readFileSync(join(projectDir, "package.json"), "utf8"));
    const link = manifest.shallot?.publish;
    if (link === undefined) return;
    if (link?.target === "itch" && typeof link.name === "string" && validItchName(link.name))
        return { target: "itch", name: link.name };
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
            'Invalid shallot.publish in package.json: expected target cloudflare, name, and accountId, or target itch and name "<user>/<game>".',
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
    /** Per file. */
    bytes: number;
    /** All files together. */
    total?: number;
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
                `${asset.path}: ${asset.size} bytes exceeds the target's ${limits.bytes} byte (${limits.bytes % 1024 ** 2 ? `${limits.bytes / 1000 / 1000} MB` : `${limits.bytes / 1024 ** 2} MiB`}) file limit.`,
            );
    }
    const total = assets.reduce((sum, asset) => sum + asset.size, 0);
    if (limits.total !== undefined && total > limits.total)
        throw new Error(
            `The web build's ${total} bytes exceed the target's ${limits.total} byte (${limits.total / 1000 / 1000} MB) total.`,
        );
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
        stdio: ["ignore", "pipe", "pipe"],
        encoding: "utf8",
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
        publishProgress(result.stdout, result.stderr);
        throw new Error(`Could not install ${dependency}.`);
    }
}

interface Deployment {
    /** The stable play link. */
    url: string;
    /** This build's own link. */
    buildUrl: string;
}

/** A target's setup runs before the build, its deploy returns once the upload is live, and its verify plays it. */
interface PublishTarget<Link extends PublishLink> {
    limits: PublishLimits;
    setup(
        projectDir: string,
        work: string,
        interactive: boolean,
        saved?: Link,
    ): Promise<{
        link: Link;
        deploy(directory: string): Promise<Deployment>;
        commands: string;
    }>;
    headers(directory: string): void;
    verify(
        browser: Browser,
        deployment: Deployment,
        directory: string,
        assets: PublishAsset[],
    ): Promise<PublishVerification>;
}

const targets: {
    cloudflare: PublishTarget<CloudflareLink>;
    itch: PublishTarget<ItchLink>;
} = {
    cloudflare: {
        limits: {
            files: 20_000,
            bytes: 25 * 1024 * 1024,
            reserved: ["_headers", "_redirects", ".assetsignore"],
        },
        headers(directory) {
            writeFileSync(join(directory, "_headers"), publishHeaders());
        },
        verify: (browser, deployment, directory, assets) =>
            verifyPublishedGame(browser, deployment.url, directory, assets),
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
            let accountId = process.env.CLOUDFLARE_ACCOUNT_ID || saved?.accountId;
            if (saved && saved.accountId !== accountId)
                throw new Error(
                    `Resolved Cloudflare account ${accountId} differs from shallot.publish.accountId ${saved.accountId}; refusing to switch accounts.`,
                );
            const identity = (): {
                loggedIn: boolean;
                accounts?: { id: string; name: string }[];
            } => {
                const result = run(["whoami", "--json"]);
                let data: { loggedIn: boolean; accounts?: { id: string; name: string }[] };
                try {
                    data = JSON.parse(result.stdout);
                } catch {
                    // whoami can fail listing accounts even when account-scoped access works.
                    if (accountId) return { loggedIn: true };
                    throw new Error(
                        `Could not determine Wrangler login.\n${result.stderr || result.stdout}`,
                    );
                }
                if (result.status !== 0 && data.loggedIn !== false) {
                    if (accountId) return { loggedIn: true };
                    throw new Error(`Wrangler whoami failed.\n${result.stderr}`);
                }
                return data;
            };
            // Wrangler 4.151.3 requireAuth/getOrSelectAccountId accepts a token and an
            // explicit account before fetchAllAccounts; ownership and deploy prove access.
            let user =
                accountId && process.env.CLOUDFLARE_API_TOKEN
                    ? { loggedIn: true, accounts: undefined }
                    : identity();
            if (!user.loggedIn) {
                if (!interactive)
                    throw new Error(
                        "Cloudflare login is missing. Set CLOUDFLARE_API_TOKEN for non-interactive publish; run wrangler login in a terminal (wrangler login --device for a remote session).",
                    );
                publishProgress(CLOUDFLARE_LOGIN_DISCLOSURE);
                if (
                    run(
                        [
                            ...CLOUDFLARE_LOGIN_ARGS,
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
            if (!accountId) {
                if (!Array.isArray(accounts) || !accounts.length)
                    throw new Error("Wrangler reported no accessible Cloudflare accounts.");
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
                if (!accounts.some((account) => account.id === accountId))
                    throw new Error(
                        `Cloudflare account ${accountId} is not accessible to this login.`,
                    );
            }
            publishProgress(
                `Using Cloudflare account ${accounts?.find((account) => account.id === accountId)?.name ?? accountId} (${accountId}).`,
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
            const link: CloudflareLink = { target: "cloudflare", name, accountId };
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
                    try {
                        return cloudflareDeployment(result);
                    } catch (error) {
                        if (!interactive) publishProgress(result.stdout, result.stderr);
                        throw error;
                    }
                },
                get commands() {
                    const config = discoverWranglerConfig(projectDir);
                    const dashboard = `Workers (all in this account): https://dash.cloudflare.com/${accountId}/workers/overview`;
                    const confirmation =
                        "Confirm with the person before deleting: Wrangler asks nothing without a TTY or in CI.";
                    if (config)
                        return `Wrangler config found: ${config}; Wrangler would take its account_id over the link's account. Use the account's Workers page instead.\n${confirmation}\n${dashboard}`;
                    const command = `CLOUDFLARE_ACCOUNT_ID=${accountId} bunx wrangler`;
                    return `Preview: ${command} dev --assets <web-build-directory>\nRollback: ${command} rollback --name ${name}\n${confirmation}\nUnpublish: ${command} delete --name ${name}\n${dashboard}`;
                },
            };
        },
    },
    itch: {
        limits: { files: 1000, bytes: 200_000_000, total: 500_000_000 },
        // itch.io serves the build with its own headers.
        headers() {},
        // itch.io's page answers a bot check to every automated client; its frame document does not.
        async verify(browser, deployment, directory, assets) {
            return verifyPublishedGame(
                browser,
                await liveItchBuild(deployment.url, deployment.buildUrl),
                directory,
                assets,
                TIMEOUT.startup,
                ITCH_HOST,
            );
        },
        async setup(_projectDir, _work, interactive, saved) {
            let name = saved?.name;
            if (!name) {
                const instruction =
                    'Create the page at https://itch.io/game/new with Kind of project HTML; shallot.publish in package.json then names it, {"target": "itch", "name": "<user>/<game>"}.';
                if (!interactive) throw new Error(`No itch.io project is linked. ${instruction}`);
                publishProgress(instruction);
                name = await ask("itch.io project (<user>/<game>)");
            }
            if (!validItchName(name))
                throw new Error(
                    `Invalid itch.io project ${JSON.stringify(name)}; use <user>/<game> as in its page's link.`,
                );
            const project = name;
            if (!interactive && !hasButlerCredentials()) throw new Error(BUTLER_LOGIN_REQUIRED);
            const butler = await ensureButler();
            const status = (timeout = BUTLER_STATUS_TIMEOUT) =>
                runButler(butler, ["status", `${project}:${ITCH_CHANNEL}`], { timeout });
            let answer = status();
            if (
                interactive &&
                answer.entries.some((entry) => /No credentials/.test(String(entry.message)))
            ) {
                publishProgress("itch.io login is required.");
                if (runButler(butler, ["login"], { terminal: true }).status !== 0)
                    throw new Error("butler login failed.");
                answer = status();
            }
            itchChannel(project, answer);
            return {
                link: { target: "itch", name: project },
                async deploy(directory) {
                    const pushed = runButler(butler, [
                        "push",
                        "--if-changed",
                        directory,
                        `${project}:${ITCH_CHANNEL}`,
                    ]);
                    const build = itchPush(project, pushed);
                    publishProgress(
                        build === undefined
                            ? "itch.io already has this build; waiting for it to be live."
                            : `Pushed build ${build}; waiting for itch.io to process it.`,
                    );
                    const live = await waitForItchBuild(project, build, status);
                    return {
                        url: itchPageUrl(project),
                        buildUrl: itchBuildUrl(live.uploadId, live.buildId),
                    };
                },
                commands: `Login: "${butler}" login\nPreview: bun run preview\nStatus: "${butler}" status ${project}\nRollback: check out an earlier commit, then bunx shallot publish --target itch\nConfirm with the person before taking the page down.\nUnpublish: https://itch.io/dashboard, Edit ${project.split("/")[1]}, set Visibility to Draft\n${itchPageSettings(project)}`,
            };
        },
    },
};

/** Publish a web build; the provider's CLI owns authentication and credential storage. */
export async function publishProject(projectDir: string, targetName?: string): Promise<void> {
    if (targetName !== undefined && !Object.hasOwn(targets, targetName))
        throw new Error(
            `Unknown publish target: ${targetName}. Available: ${Object.keys(targets).join(", ")}.`,
        );
    const linked = readPublishLink(projectDir);
    const chosen = (targetName ?? linked?.target ?? "cloudflare") as keyof typeof targets;
    const saved = linked?.target === chosen ? linked : undefined;
    const target = targets[chosen] as PublishTarget<PublishLink>;
    if (!targetName && !linked) publishProgress("Using publish target cloudflare.");
    const work = mkdtempSync(join(tmpdir(), "shallot-publish-"));
    let playUrl: string;
    try {
        const directory = join(work, "web");
        const setup = await target.setup(
            projectDir,
            work,
            publishInteraction(process.env, !!process.stdin.isTTY && !!process.stdout.isTTY),
            saved,
        );
        await buildWeb(projectDir, { outDir: directory, stdout: "quiet" });
        const assets = publishAssets(directory);
        preflightPublish(assets, target.limits);
        target.headers(directory);
        const deployed = await setup.deploy(directory);
        savePublishLink(projectDir, setup.link);
        publishProgress(`Build: ${deployed.buildUrl}\n${setup.commands}`);
        const browser = await projectChromium(projectDir);
        try {
            const verification = await target.verify(browser, deployed, directory, assets);
            publishProgress(
                `${verification.hardware ? "Engine started on a hardware adapter." : "Engine started; loading, headers and assets verified, not hardware play."}${verification.threads === undefined ? "" : ` Physics ${verification.physicsStepped ? "stepped" : "initialized"} on ${verification.threads} threads.`}\nAdapter: ${verification.adapter}`,
            );
        } finally {
            await browser.close();
        }
        playUrl = deployed.url;
    } finally {
        rmSync(work, { recursive: true, force: true });
    }
    console.log(playUrl);
}
