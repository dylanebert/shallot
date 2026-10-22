import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import {
    mkdirSync,
    mkdtempSync,
    readFileSync,
    statSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "../../scripts/bun-child";
import { publishTestChromium } from "../../scripts/publish-browser";
import { publishTestButler } from "../../scripts/publish-butler";
import { removeTestDirectory } from "../../scripts/test-directory";
import { buildWeb, contentType } from "../project";
import { publishAssets } from "./publish";
import {
    BUTLER_VERSION,
    butlerPlatform,
    ITCH_HOST,
    ITCH_HTML_SCRIPT,
    runButler,
    waitForItchBuild,
} from "./publish-itch";
import { verifyPublishedGame } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const repo = resolve(import.meta.dir, "../..");
const recorded = JSON.parse(
    readFileSync(join(import.meta.dir, "fixtures/publish-itch.json"), "utf8"),
);
const headers = recorded.cdnHeaders as Record<
    "document" | "asset" | "missing",
    Record<string, string>
>;
const root = mkdtempSync(join(tmpdir(), "shallot-publish-itch-node-"));
const build = "/html/19938966-2182067/";
const project = "individualkex/shallot-publish-test";
let directory = "";
let browser: Awaited<ReturnType<typeof publishTestChromium>>;

/**
 * itch.io's HTML CDN as recorded: the build under /html/<upload>-<build>/, its index.html ending in itch.io's
 * script, isolated with SharedArrayBuffer support on and without isolation headers when it is off.
 */
function cdn(isolated: boolean) {
    return Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch(request) {
            const { pathname } = new URL(request.url);
            const path = pathname.startsWith(build) ? pathname.slice(build.length) : "";
            const file = join(directory, path);
            if (
                !path ||
                path.includes("..") ||
                !statSync(file, { throwIfNoEntry: false })?.isFile()
            )
                return new Response("missing", {
                    status: 404,
                    headers: isolated ? headers.missing : {},
                });
            const type =
                (
                    {
                        html: "text/html; charset=utf-8",
                        js: "text/javascript; charset=utf-8",
                        css: "text/css",
                        woff2: "font/woff2",
                        txt: "text/plain",
                    } as Record<string, string>
                )[path.split(".").pop()!] ??
                contentType(path) ??
                "application/octet-stream";
            const answered = path === "index.html" ? headers.document : headers.asset;
            const body = readFileSync(file);
            return new Response(
                path === "index.html" ? Buffer.concat([body, Buffer.from(ITCH_HTML_SCRIPT)]) : body,
                {
                    headers: {
                        ...(isolated ? answered : { "Cache-Control": answered["Cache-Control"]! }),
                        "Content-Type": type,
                    },
                },
            );
        },
    });
}
const isolatedCdn = cdn(true);
const classicCdn = cdn(false);
const builds: Record<string, string> = {};

beforeAll(async () => {
    browser = await publishTestChromium();
    // itch.io's script as recorded, and its beacon refused as html-callback's CORP refused it, without leaving the host.
    const newContext = browser.newContext.bind(browser);
    browser.newContext = async (...args) => {
        const context = await newContext(...args);
        // The last matching route handles a request, so this refusal of every other host goes first.
        await context.route(/^https?:\/\/(?!127\.0\.0\.1)/, (route) => route.abort());
        await context.route("https://static.itch.io/htmlgame.js", (route) =>
            route.fulfill({ body: recorded.htmlgameJs, headers: recorded.htmlgameHeaders }),
        );
        await context.route("https://itch.io/html-callback", (route) =>
            route.abort("blockedbyresponse"),
        );
        return context;
    };
    for (const example of ["minimal", "physics"]) {
        builds[example] = join(root, example);
        await buildWeb(join(repo, "examples", example), { outDir: builds[example], stdout: 2 });
    }
});
afterAll(async () => {
    await browser?.close();
    await Promise.all([isolatedCdn, classicCdn].map((server) => server.stop(true)));
    await removeTestDirectory(root);
});

const verify = (server: { url: URL }, path = build) =>
    verifyPublishedGame(
        browser,
        `${server.url.origin}${path}index.html?v=1792635744`,
        directory,
        publishAssets(directory),
        TIMEOUT.startup,
        ITCH_HOST,
        TIMEOUT.cheap,
    );

for (const example of ["minimal", "physics"])
    test(`the verifier plays ${example} from the frame document on the recorded HTML CDN`, async () => {
        directory = builds[example]!;
        const context = await browser.newContext();
        const errors: string[] = [];
        const visit = await context.newPage();
        const logged = visit.waitForEvent("console", {
            predicate: (message) => message.type() === "error",
        });
        visit.on("console", (message) => {
            if (message.type() === "error")
                errors.push(`${message.location().url} ${message.text()}`);
        });
        await visit.goto(`${isolatedCdn.url.origin}${build}index.html?v=1792635744`);
        await logged;
        await context.close();
        expect(errors).toEqual([
            "https://itch.io/html-callback Failed to load resource: net::ERR_BLOCKED_BY_RESPONSE",
        ]);
        const verification = await verify(isolatedCdn);
        if (example === "physics") expect(verification.threads).toBeGreaterThan(1);
        else expect(verification.threads).toBeUndefined();
    });

test("the frame document fails verification without SharedArrayBuffer support or before it is served", async () => {
    directory = builds.minimal!;
    for (const [failure, attempt] of [
        ["missing Cross-Origin-Embedder-Policy: require-corp", () => verify(classicCdn)],
        [": expected 200, got 404", () => verify(isolatedCdn, "/html/19938966-2182534/")],
    ] as const)
        expect(
            await attempt().then(
                () => "verified",
                (error: Error) => error.message,
            ),
        ).toContain(failure);
});

// A stall short of the tier, so the bound and not the tier stops it.
const stall = TIMEOUT.cheap;

test("a stuck butler is stopped, and the processing wait is bounded with it", async () => {
    const stuck = join(root, "stuck-butler.ts");
    writeFileSync(stuck, "await Bun.sleep(60_000);\n");
    const started = Date.now();
    expect(() => runButler(process.execPath, [stuck], { timeout: stall })).toThrow(
        `did not answer within ${Math.ceil(stall / 1000)} s`,
    );
    const limits: number[] = [];
    await expect(
        waitForItchBuild(
            project,
            2182067,
            (limit) => {
                limits.push(limit);
                return runButler(process.execPath, [stuck], { timeout: stall });
            },
            stall,
        ),
    ).rejects.toThrow("did not answer");
    expect(limits.length).toBe(1);
    expect(limits[0]!).toBeLessThanOrEqual(stall);
    expect(Date.now() - started).toBeLessThan(TIMEOUT.node);
});

test("the pinned butler dry-runs a push of the build without credentials", () => {
    directory = builds.minimal!;
    const butler = publishTestButler();
    const home = mkdtempSync(join(root, "home-"));
    const env = {
        PATH: process.env.PATH,
        HOME: home,
        XDG_CONFIG_HOME: home,
        USERPROFILE: home,
        APPDATA: home,
    };
    const version = Bun.spawnSync([butler, "version"], { env, stdout: "pipe", stderr: "pipe" });
    expect(`${version.stdout}${version.stderr}`).toContain(`v${BUTLER_VERSION},`);
    const dry = Bun.spawnSync(
        [butler, "--json", "push", "--dry-run", directory, `${project}:html5`],
        {
            env,
            stdin: "ignore",
            stdout: "pipe",
            stderr: "pipe",
        },
    );
    expect(dry.exitCode, `${dry.stdout}\n${dry.stderr}`).toBe(0);
    const result = dry.stdout
        .toString()
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line))
        .findLast((entry) => entry.type === "result");
    expect(result.value).toMatchObject({ dryRun: true, reason: "dry-run", channel: "html5" });
    expect(dry.stdout.toString()).toContain(`(${publishAssets(directory).length} files,`);
});

test("a linked itch publish with no terminal and no BUTLER_API_KEY fails naming it, without prompting", () => {
    const butler = publishTestButler();
    const home = mkdtempSync(join(root, "home-"));
    const cache = join(home, "cache");
    // The child's clean cache holds the declared butler, so publish fetches nothing.
    const cached = join(cache, "shallot", "butler", BUTLER_VERSION, butlerPlatform());
    mkdirSync(dirname(cached), { recursive: true });
    symlinkSync(dirname(butler), cached, "junction");
    const game = join(home, "game");
    mkdirSync(game);
    const manifest = JSON.stringify({
        name: "game",
        shallot: { publish: { target: "itch", name: project } },
    });
    writeFileSync(join(game, "package.json"), manifest);
    const env = {
        ...Object.fromEntries(
            Object.entries(process.env).filter(([name]) => name !== "BUTLER_API_KEY"),
        ),
        HOME: home,
        XDG_CONFIG_HOME: home,
        USERPROFILE: home,
        APPDATA: home,
        XDG_CACHE_HOME: cache,
        LOCALAPPDATA: cache,
        CI: "true",
    };
    const refusal = Bun.spawnSync({
        ...bunChild([join(repo, "bin/shallot.ts"), "publish"]),
        cwd: game,
        env,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
        timeout: TIMEOUT.startup,
    });
    expect(refusal.exitCode, refusal.stderr.toString()).toBe(1);
    expect(refusal.stderr.toString()).toContain("itch.io login is missing. Set BUTLER_API_KEY");
    expect(refusal.stderr.toString()).not.toContain("Fetching");
    expect(refusal.stdout.toString()).toBe("");
    expect(readFileSync(join(game, "package.json"), "utf8")).toBe(manifest);
});
