import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { inflateRawSync } from "node:zlib";
import { CROSS_ORIGIN_ISOLATION } from "../project";
import { publishProgress } from "./publish-output";
import type { PublishHost } from "./publish-verify";

export const BUTLER_VERSION = "15.32.0";
/** SHA-256 recorded at download of broth's `default` archive per platform, not publisher attestations. */
export const BUTLER_ARCHIVES: Record<string, string> = {
    "darwin-arm64": "f1c460f08c02686b49f8da74344e5578ece9ea1a8bae1779445b5d98e0a794b3",
    "darwin-amd64": "a92947c9556c9bfb0cc9bdc3814f342b30e0ee7269efc418658894a916ca9584",
    "linux-amd64": "2335971394ef6596f95ded0833e85ee28755e13761716ef4c91d6b11f69162f5",
    "linux-arm64": "4f2da37785fd9cbc15052427490e4e4f5cd735a019f21554d6273fa9a3490152",
    "windows-amd64": "5902c488b86bb3518ca00bf4e706d7687f229e3f0f28653cd943348ea9ec8878",
};
const BROTH = "https://broth.itch.zone/butler";
export const ITCH_CHANNEL = "html5";

export interface ItchLink {
    target: "itch";
    /** itch.io's `<user>/<game>`, as butler names a project. */
    name: string;
}

export function validItchName(name: string): boolean {
    return /^[a-z0-9_-]+\/[a-z0-9_-]+$/i.test(name);
}

export function itchPageUrl(name: string): string {
    const [user, game] = name.split("/");
    return `https://${user!.toLowerCase()}.itch.io/${game}`;
}

export function butlerPlatform(
    platform: NodeJS.Platform = process.platform,
    arch: string = process.arch,
): string {
    const os = { darwin: "darwin", linux: "linux", win32: "windows" }[platform as string];
    const cpu = { arm64: "arm64", x64: "amd64" }[arch];
    const name = `${os}-${cpu}`;
    if (!os || !cpu || !BUTLER_ARCHIVES[name])
        throw new Error(
            `itch.io's butler ${BUTLER_VERSION} has no build for ${platform}-${arch}. Available: ${Object.keys(BUTLER_ARCHIVES).join(", ")}.`,
        );
    return name;
}

/** The pinned butler's executable in Shallot's cache (XDG_CACHE_HOME-aware; LOCALAPPDATA on Windows). */
export function butlerPath(platform = butlerPlatform()): string {
    const cache =
        process.platform === "win32"
            ? (process.env.LOCALAPPDATA ?? resolve(homedir(), "AppData", "Local"))
            : (process.env.XDG_CACHE_HOME ?? resolve(homedir(), ".cache"));
    return join(
        cache,
        "shallot",
        "butler",
        BUTLER_VERSION,
        platform,
        platform.startsWith("windows") ? "butler.exe" : "butler",
    );
}

/** Extract a zip's regular files, stored or deflated, keeping their Unix modes. */
export function extractZip(archive: Buffer, directory: string): void {
    const end = archive.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
    if (end < 0) throw new Error("not a zip archive");
    let offset = archive.readUInt32LE(end + 16);
    for (let index = archive.readUInt16LE(end + 10); index > 0; index--) {
        if (archive.readUInt32LE(offset) !== 0x02014b50) throw new Error("corrupt zip directory");
        const method = archive.readUInt16LE(offset + 10);
        const size = archive.readUInt32LE(offset + 20);
        const nameLength = archive.readUInt16LE(offset + 28);
        const extraLength = archive.readUInt16LE(offset + 30);
        const commentLength = archive.readUInt16LE(offset + 32);
        const mode = archive.readUInt32LE(offset + 38) >>> 16;
        const local = archive.readUInt32LE(offset + 42);
        const name = archive.toString("utf8", offset + 46, offset + 46 + nameLength);
        offset += 46 + nameLength + extraLength + commentLength;
        if (name.endsWith("/")) continue;
        if (name.includes("..") || name.startsWith("/") || name.includes("\\"))
            throw new Error(`zip entry escapes its directory: ${name}`);
        const start =
            local + 30 + archive.readUInt16LE(local + 26) + archive.readUInt16LE(local + 28);
        const data = archive.subarray(start, start + size);
        if (method !== 0 && method !== 8) throw new Error(`unsupported zip method ${method}`);
        const path = join(directory, name);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, method === 8 ? inflateRawSync(data) : data);
        if (mode) chmodSync(path, mode & 0o777);
    }
}

/** Fetch the pinned butler into Shallot's cache unless it is already there; refuses a changed archive. */
export async function ensureButler(
    platform = butlerPlatform(),
    source = `${BROTH}/${platform}/${BUTLER_VERSION}/archive/default`,
): Promise<string> {
    const path = butlerPath(platform);
    const complete = join(dirname(path), ".complete");
    if (existsSync(complete)) return path;
    publishProgress(`Fetching itch.io's butler ${BUTLER_VERSION} for ${platform}.`);
    const response = await fetch(source);
    if (!response.ok)
        throw new Error(
            `Could not fetch butler ${BUTLER_VERSION}: ${source} answered ${response.status}.`,
        );
    const archive = Buffer.from(await response.arrayBuffer());
    const hash = createHash("sha256").update(archive).digest("hex");
    if (hash !== BUTLER_ARCHIVES[platform])
        throw new Error(
            `butler ${BUTLER_VERSION} for ${platform} from ${source} has SHA-256 ${hash}, not the pinned ${BUTLER_ARCHIVES[platform]}; refusing to run it.`,
        );
    // Each fetch stages apart; a concurrent fetch that installed first is kept.
    const staging = `${dirname(path)}.partial-${process.pid}-${Date.now()}`;
    try {
        extractZip(archive, staging);
        writeFileSync(join(staging, ".complete"), "");
        if (existsSync(complete)) return path;
        rmSync(dirname(path), { recursive: true, force: true });
        renameSync(staging, dirname(path));
    } catch (error) {
        if (!existsSync(complete)) throw error;
    } finally {
        rmSync(staging, { recursive: true, force: true });
    }
    return path;
}

export interface ButlerResult {
    status: number | null;
    entries: Record<string, unknown>[];
    stderr: string;
}

/** A butler status answers in seconds; past this it is stuck. */
export const BUTLER_STATUS_TIMEOUT = 60_000;

/**
 * Runs butler with `--json`, or with the terminal's stdio for `terminal`. `timeout` in milliseconds stops a
 * child that has not exited and refuses; a push is unbounded, since its upload time follows the build's size.
 */
export function runButler(
    butler: string,
    args: string[],
    { terminal = false, timeout }: { terminal?: boolean; timeout?: number } = {},
): ButlerResult {
    const result = spawnSync(butler, terminal ? args : ["--json", ...args], {
        encoding: "utf8",
        stdio: terminal ? "inherit" : ["ignore", "pipe", "pipe"],
        timeout,
        killSignal: "SIGKILL",
    });
    if ((result.error as NodeJS.ErrnoException | undefined)?.code === "ETIMEDOUT")
        throw new Error(`butler ${args[0]} did not answer within ${Math.ceil(timeout! / 1000)} s.`);
    if (result.error) throw result.error;
    return {
        status: result.status,
        stderr: result.stderr ?? "",
        entries: (result.stdout ?? "")
            .split("\n")
            .filter((line) => line.startsWith("{"))
            .map((line) => JSON.parse(line)),
    };
}

function failure(result: ButlerResult): string | undefined {
    if (result.status === 0) return;
    const error = result.entries.findLast((entry) => entry.type === "error")?.message;
    return String(error ?? result.stderr).split("\n")[0];
}

export const MISSING_BUTLER_LOGIN = "No credentials and stdin is not a terminal";

export interface ItchChannel {
    uploadId: number;
    head?: { id: number; state: string };
    pending?: { id: number; state: string };
}

/** Refuses butler's answers for a missing login and a missing project by what to do about them. */
function refuseButlerFailure(name: string, command: string, result: ButlerResult): void {
    const message = failure(result);
    if (message?.includes(MISSING_BUTLER_LOGIN))
        throw new Error(
            "itch.io login is missing. Set BUTLER_API_KEY for non-interactive publish (https://itch.io/user/settings/api-keys); run butler login in a terminal.",
        );
    if (message?.includes("invalid game") || message?.includes("invalid target"))
        throw new Error(
            `itch.io has no project ${name} for this login. Create it at https://itch.io/game/new with Kind of project HTML, then publish again; package.json's shallot.publish.name names it.`,
        );
    if (message) throw new Error(`butler ${command} failed: ${message}`);
}

/** Reads `butler status <name>:html5`: undefined until the channel's first build is processed. */
export function itchChannel(name: string, result: ButlerResult): ItchChannel | undefined {
    refuseButlerFailure(name, "status", result);
    const value = result.entries.findLast((entry) => entry.type === "result")?.value as
        | { channels?: (ItchChannel & { name: string })[] }
        | undefined;
    if (!Array.isArray(value?.channels)) throw new Error("butler status returned no channel list.");
    return value.channels.find((channel) => channel.name === ITCH_CHANNEL);
}

/** Reads `butler push`: the new build's id, or undefined when `--if-changed` skipped an identical build. */
export function itchPush(name: string, result: ButlerResult): number | undefined {
    refuseButlerFailure(name, "push", result);
    const value = result.entries.findLast((entry) => entry.type === "result")?.value as
        | { buildId?: number; skipped?: boolean }
        | undefined;
    if (value?.skipped) return;
    if (typeof value?.buildId !== "number" || value.buildId <= 0)
        throw new Error("butler push reported no build.");
    return value.buildId;
}

export type ItchBuildState =
    | { done: true; uploadId: number; buildId: number }
    | { done: false; state: string };

/** Whether `build` is the channel's processed head; a failed build refuses. `build` undefined waits for any head. */
export function itchBuildState(channel: ItchChannel | undefined, build?: number): ItchBuildState {
    const pending = channel?.pending;
    if (pending && (build === undefined || pending.id === build) && pending.state === "failed")
        throw new Error(`itch.io failed to process build ${pending.id}; see butler status.`);
    const head = channel?.head;
    if (head && (build === undefined || head.id === build)) {
        if (head.state === "failed")
            throw new Error(`itch.io failed to process build ${head.id}; see butler status.`);
        if (head.state === "completed" && !(build === undefined && pending))
            return { done: true, uploadId: channel!.uploadId, buildId: head.id };
    }
    return { done: false, state: pending?.state ?? head?.state ?? "queued" };
}

/**
 * Polls `status` every `interval` until `build` is the channel's processed head, refusing after `wait`
 * milliseconds; each status call is given at most the time left. `build` undefined waits for any head.
 */
export async function waitForItchBuild(
    name: string,
    build: number | undefined,
    status: (timeout: number) => ButlerResult,
    wait = 10 * 60_000,
    interval = 2000,
): Promise<{ uploadId: number; buildId: number }> {
    const deadline = Date.now() + wait;
    while (true) {
        const left = deadline - Date.now();
        const state = itchBuildState(
            itchChannel(name, status(Math.max(1, Math.min(BUTLER_STATUS_TIMEOUT, left)))),
            build,
        );
        if (state.done) return state;
        if (Date.now() + interval > deadline)
            throw new Error(
                `itch.io has not processed build ${build ?? "head"} after ${Math.round(wait / 60_000)} minutes (${state.state}); see butler status ${name}.`,
            );
        await new Promise((done) => setTimeout(done, interval));
    }
}

/** The document itch.io's embed frames for a build, on its HTML CDN. */
export function itchBuildUrl(uploadId: number, buildId: number): string {
    return `https://html.itch.zone/html/${uploadId}-${buildId}/index.html`;
}

/** itch.io's script, which its HTML CDN appends to a build's index.html. */
export const ITCH_HTML_SCRIPT =
    '<script defer src="https://static.itch.io/htmlgame.js" type="text/javascript"></script>';

/**
 * How itch.io's HTML CDN serves a build: it isolates the frame document and appends its script to it, and
 * the build's other files carry COEP alone. The script's beacon to itch.io's html-callback is refused by
 * its response's CORP and logs a console error, which is itch.io's, not the game's.
 */
export const ITCH_HOST: PublishHost = {
    headers: (path) =>
        path === "index.html"
            ? CROSS_ORIGIN_ISOLATION
            : {
                  "Cross-Origin-Embedder-Policy":
                      CROSS_ORIGIN_ISOLATION["Cross-Origin-Embedder-Policy"],
              },
    bytes: (path, built) =>
        path === "index.html" ? Buffer.concat([built, Buffer.from(ITCH_HTML_SCRIPT)]) : built,
    foreign: (url) => url === "https://itch.io/html-callback",
};

/** The page settings only itch.io's web UI shows, which verifying the frame document cannot see. */
export function itchPageSettings(name: string): string {
    return `Not verified, set on itch.io: on https://itch.io/dashboard, Edit ${name.split("/")[1]}; tick "This file will be played in the browser" on the ${ITCH_CHANNEL} upload and "SharedArrayBuffer support" under Embed options, and set Visibility to Public or Restricted, not Draft. Without them ${itchPageUrl(name)} offers no game, one without threads, or nothing to a visitor.`;
}

/**
 * itch.io puts a processed build on its HTML CDN only once its page is viewed, and then removes the build it
 * replaces; until then the frame document answers 404, which the CDN's edge caches for 31 days. So the
 * document is read with a fresh query, as the page frames it with `?v=`, and the same URL is verified.
 * Refuses a build not yet live.
 */
export async function liveItchBuild(pageUrl: string, buildUrl: string): Promise<string> {
    const url = `${buildUrl}?v=${Date.now()}`;
    const response = await fetch(url);
    await response.body?.cancel();
    if (response.status === 404)
        throw new Error(
            `Not verified yet: itch.io serves this build once its page is viewed. Open ${pageUrl}, then run shallot publish again to verify ${buildUrl}.`,
        );
    return url;
}
