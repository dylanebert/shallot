import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseCliArgs } from "./index";
import {
    preflightPublish,
    publishHeaders,
    publishInteraction,
    readPublishLink,
    savePublishLink,
} from "./publish";
import {
    cloudflareConfig,
    cloudflareDeployment,
    type WranglerResult,
    workerExists,
} from "./publish-cloudflare";

const root = mkdtempSync(join(tmpdir(), "shallot-publish-unit-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
const answer = (
    status: number,
    stdout = "",
    entries: WranglerResult["entries"] = [],
): WranglerResult => ({ status, stdout, entries, stderr: "provider failure" });

test("publish takes only the target option and has command help", () => {
    expect(parseCliArgs(["publish"])).toEqual({ kind: "publish" });
    expect(parseCliArgs(["publish", "--target", "cloudflare"])).toEqual({
        kind: "publish",
        target: "cloudflare",
    });
    expect(parseCliArgs(["publish", "--target=cloudflare"])).toEqual({
        kind: "publish",
        target: "cloudflare",
    });
    expect(parseCliArgs(["help", "publish"])).toEqual({ kind: "command-help", command: "publish" });
    for (const args of [
        ["--target"],
        ["--target="],
        ["--target", "--helpful"],
        ["game"],
        ["--temporary"],
        ["--target", "cloudflare", "extra"],
    ])
        expect(() => parseCliArgs(["publish", ...args])).toThrow("publish accepts only");
});

test("the link has one home, preserves other Shallot settings, and rejects malformed links", () => {
    const path = join(root, "package.json");
    writeFileSync(path, JSON.stringify({ name: "game", shallot: { other: 12 } }));
    expect(readPublishLink(root)).toBeUndefined();
    const link = { target: "cloudflare" as const, name: "game", accountId: "a".repeat(32) };
    savePublishLink(root, link);
    expect(readPublishLink(root)).toEqual(link);
    expect(JSON.parse(readFileSync(path, "utf8")).shallot.other).toBe(12);
    for (const invalid of [
        null,
        {},
        { ...link, name: "../game" },
        { ...link, target: "unknown" },
        { ...link, accountId: "bad" },
        { target: "cloudflare", name: "game" },
    ]) {
        writeFileSync(path, JSON.stringify({ shallot: { publish: invalid } }));
        expect(() => readPublishLink(root)).toThrow("Invalid shallot.publish");
    }
});

// Recorded shapes: Wrangler 4.151.3 versions/list.ts, register-yargs-command.ts, and
// deploy-helpers/deploy/helpers/worker-not-found-error.ts. No account was contacted.
test("only the structured Worker-not-found answer permits first-link creation", () => {
    expect(workerExists(answer(0, "[]"))).toBe(true);
    expect(
        workerExists(answer(0, '[{"id":"version","metadata":{"created_on":"2026-10-21"}}]')),
    ).toBe(true);
    expect(workerExists(answer(1, "", [{ type: "command-failed", version: 1, code: 10007 }]))).toBe(
        false,
    );
    for (const code of [undefined, 10000, 10001, 10002, 9109, 429, 500])
        expect(() =>
            workerExists(answer(1, "", [{ type: "command-failed", version: 1, code }])),
        ).toThrow("Could not check Worker ownership");
    expect(() => workerExists(answer(1, "Worker not found [code: 10007]"))).toThrow();
    expect(() => workerExists(answer(0, "{}"))).toThrow();
    expect(() => workerExists(answer(0, "not JSON"))).toThrow();
});

test("no TTY, CI and agent sessions cannot prompt", () => {
    expect(publishInteraction({}, false)).toBe(false);
    expect(publishInteraction({ CI: "true" }, true)).toBe(false);
    for (const env of [
        { CLAUDECODE: "1" },
        { CODEX_THREAD_ID: "t" },
        { PI_CODING_AGENT: "true" },
        { AGENT: "amp" },
        { CURSOR_TRACE_ID: "t", PAGER: "head -n 10000 | cat" },
        { TERM_PROGRAM: "zed", PAGER: "cat" },
        { TERM_PROGRAM: "vscode", GIT_PAGER: "cat" },
    ])
        expect(publishInteraction(env, true)).toBe(false);
});

test("a person in an editor's terminal can prompt", () => {
    for (const env of [
        {},
        { CI: "false" },
        { CURSOR_TRACE_ID: "t" },
        { TERM_PROGRAM: "zed" },
        { TERM_PROGRAM: "vscode" },
        { TERM_PROGRAM: "WarpTerminal" },
    ])
        expect(publishInteraction(env, true)).toBe(true);
});

test("preflight names the file and limit, including the first file over the count", () => {
    const index = { path: "index.html", size: 1 };
    expect(() =>
        preflightPublish([index, { path: "huge.wasm", size: 26 * 1024 * 1024 }], {
            files: 20000,
            bytes: 25 * 1024 * 1024,
        }),
    ).toThrow("huge.wasm: 27262976 bytes exceeds the target's 26214400 byte (25 MiB)");
    expect(() =>
        preflightPublish([index, { path: "extra.js", size: 1 }], { files: 1, bytes: 1 }),
    ).toThrow("extra.js: exceeds the target's 1 file limit");
    for (const path of ["_headers", "_redirects", ".assetsignore"])
        expect(() =>
            preflightPublish([index, { path, size: 0 }], {
                files: 10,
                bytes: 10,
                reserved: [path],
            }),
        ).toThrow(`${path}: reserved`);
    expect(() =>
        preflightPublish([index, { path: "_headers", size: 0 }], { files: 10, bytes: 10 }),
    ).not.toThrow();
    expect(() => preflightPublish([], { files: 10, bytes: 10 })).toThrow("no index.html");
    expect(() =>
        preflightPublish([{ path: "index.html", size: 25 * 1024 * 1024 }], {
            files: 1,
            bytes: 25 * 1024 * 1024,
        }),
    ).not.toThrow();
});

test("the generated configuration is assets-only and headers isolate and cache the build", () => {
    const config = cloudflareConfig("/web", {
        target: "cloudflare",
        name: "game",
        accountId: "a".repeat(32),
    });
    expect(config).not.toHaveProperty("main");
    expect(config.assets.directory).toBe("/web");
    expect(config.assets.not_found_handling).toBe("none");
    expect(publishHeaders()).toContain("Cross-Origin-Opener-Policy: same-origin");
    expect(publishHeaders()).toContain("Cross-Origin-Embedder-Policy: require-corp");
    expect(publishHeaders()).toContain(
        "/assets/*\n  Cache-Control: public, max-age=31536000, immutable",
    );
    expect(publishHeaders()).toContain("/index.html\n  Cache-Control: no-cache");
});

test("deployment reporting consumes Wrangler's structured result, not its prose", () => {
    expect(
        cloudflareDeployment(
            answer(
                0,
                "",
                JSON.parse(
                    '[{"type":"deploy","version_id":"12345678-rest","targets":["https://game.user.workers.dev"]}]',
                ),
            ),
        ),
    ).toEqual({
        url: "https://game.user.workers.dev",
        buildUrl: "https://12345678-game.user.workers.dev",
    });
    expect(() => cloudflareDeployment(answer(1))).toThrow("Wrangler deploy failed");
    expect(() => cloudflareDeployment(answer(0, "https://game.user.workers.dev"))).toThrow(
        "did not report",
    );
});
