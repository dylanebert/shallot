import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { publishProgress } from "./publish-output";

/** Matches pinned Wrangler's findWranglerConfig: each filename searches every ancestor. */
export function discoverWranglerConfig(projectDir: string): string | undefined {
    for (const name of ["wrangler.json", "wrangler.jsonc", "wrangler.toml"]) {
        let directory = resolve(projectDir);
        while (true) {
            const path = join(directory, name);
            if (existsSync(path)) return path;
            const parent = dirname(directory);
            if (parent === directory) break;
            directory = parent;
        }
    }
}

export const WRANGLER_VERSION: string = JSON.parse(
    readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
).peerDependencies.wrangler;
// Wrangler packages/wrangler/src/user/commands.ts:38–40,145–167 accepts --scopes;
// Wrangler adds offline_access itself.
export const CLOUDFLARE_LOGIN_ARGS = [
    "login",
    "--scopes",
    "account:read",
    "user:read",
    "workers_scripts:write",
];
export const CLOUDFLARE_LOGIN_DISCLOSURE =
    "Cloudflare login requests account:read, user:read and workers_scripts:write; Wrangler adds offline_access. This is Wrangler's own login, shared with other Wrangler use. Other work may need wrangler login again. For a remote session, use wrangler login --device.";

export interface CloudflareLink {
    target: "cloudflare";
    name: string;
    accountId: string;
}
export interface WranglerResult {
    status: number | null;
    stdout: string;
    stderr: string;
    entries: Record<string, unknown>[];
}

// biome-ignore-start lint/style/useNamingConvention: Wrangler owns its configuration field names.
export function cloudflareConfig(directory: string, link: CloudflareLink) {
    return {
        name: link.name,
        account_id: link.accountId,
        compatibility_date: "2026-10-19",
        workers_dev: true,
        preview_urls: true,
        assets: { directory, not_found_handling: "none" },
    };
}

// biome-ignore-end lint/style/useNamingConvention: Wrangler owns its configuration field names.

export function resolveWrangler(projectDir: string): string | undefined {
    const require = createRequire(join(projectDir, "package.json"));
    // An inline install must be visible even after Bun cached a missing resolution.
    const path = require.resolve
        .paths("wrangler")
        ?.map((directory) => join(directory, "wrangler/package.json"))
        .find((candidate) => existsSync(candidate));
    if (!path) return;
    const manifest = JSON.parse(readFileSync(path, "utf8"));
    if (manifest.version !== WRANGLER_VERSION)
        throw new Error(
            `publish requires wrangler ${WRANGLER_VERSION}; run bun add --dev wrangler@${WRANGLER_VERSION}`,
        );
    return resolve(dirname(path), manifest.bin.wrangler);
}

export function runWrangler(
    cwd: string,
    cli: string,
    args: string[],
    outputPath: string,
    interactive = false,
): WranglerResult {
    const quietDeploy = interactive && args[0] === "deploy";
    const logPath = `${outputPath}.log`;
    const result = spawnSync("node", [cli, ...args], {
        cwd,
        env: {
            ...process.env,
            WRANGLER_SEND_METRICS: "false",
            WRANGLER_LOG_SANITIZE: "true",
            WRANGLER_OUTPUT_FILE_PATH: outputPath,
            ...(quietDeploy ? { WRANGLER_LOG: "warn", WRANGLER_LOG_PATH: logPath } : {}),
        },
        encoding: "utf8",
        stdio: interactive ? ["inherit", 2, 2] : ["ignore", "pipe", "pipe"],
    });
    if (result.error) {
        if ((result.error as NodeJS.ErrnoException).code === "ENOENT")
            throw new Error(
                "Cloudflare publishing requires Node.js 22 or newer on PATH to run Wrangler.",
            );
        throw result.error;
    }
    // Keep the terminal for provider prompts; Wrangler's debug log retains suppressed deploy output.
    if (quietDeploy && result.status !== 0 && existsSync(logPath))
        publishProgress(readFileSync(logPath, "utf8"));
    const entries = existsSync(outputPath)
        ? readFileSync(outputPath, "utf8")
              .trim()
              .split("\n")
              .filter(Boolean)
              .map((line) => JSON.parse(line))
        : [];
    return {
        status: result.status,
        stdout: result.stdout ?? "",
        stderr: result.stderr ?? "",
        entries,
    };
}

/** Only Wrangler's Worker-not-found code authorizes creation; all other failures are inconclusive. */
export function workerExists(result: WranglerResult): boolean {
    if (result.status === 0) {
        if (!Array.isArray(JSON.parse(result.stdout)))
            throw new Error("Wrangler returned an invalid versions list.");
        return true;
    }
    const failure = result.entries.findLast((entry) => entry.type === "command-failed");
    if (failure?.code === 10007) return false;
    throw new Error(
        `Could not check Worker ownership; publish stopped.\n${result.stderr || result.stdout}`,
    );
}

export function cloudflareDeployment(result: WranglerResult): { url: string; buildUrl: string } {
    if (result.status !== 0)
        throw new Error(`Wrangler deploy failed.\n${result.stderr || result.stdout}`);
    const deployment = result.entries.findLast((entry) => entry.type === "deploy");
    const targets = deployment?.targets;
    const url =
        Array.isArray(targets) &&
        targets.find(
            (value) =>
                typeof value === "string" && /^https:\/\/[a-z0-9.-]+\.workers\.dev\/?$/.test(value),
        );
    const version = deployment?.version_id;
    if (!url || typeof version !== "string")
        throw new Error(
            "Wrangler did not report a workers.dev link and version; publish could not verify the deployment.",
        );
    return { url, buildUrl: url.replace("https://", `https://${version.slice(0, 8)}-`) };
}
