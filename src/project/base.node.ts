import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { publishTestChromium } from "../../scripts/publish-browser";
import { removeTestDirectory } from "../../scripts/test-directory";
import { publishAssets } from "../cli/publish";
import { verifyPublishedGame } from "../cli/publish-verify";
import { buildWeb } from "./build";
import { contentType } from "./assets.js";
import { CROSS_ORIGIN_ISOLATION } from "./isolation.js";

setDefaultTimeout(TIMEOUT.node);
const repo = resolve(import.meta.dir, "../..");
const root = mkdtempSync(join(tmpdir(), "shallot-base-"));
const subpath = "/html/19938966-2182067/";
let browser: Awaited<ReturnType<typeof publishTestChromium>>;
let directory = "";
// One origin per base: a build that names its files from the root fails under the subpath.
const servers = ["/", subpath].map((base) =>
    Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch(request) {
            const path = new URL(request.url).pathname;
            const relative = path.startsWith(base) ? path.slice(base.length) || "index.html" : "..";
            const file = join(directory, relative);
            if (relative.includes("..") || !statSync(file, { throwIfNoEntry: false })?.isFile())
                return new Response("missing", { status: 404 });
            const type =
                ({ html: "text/html", js: "text/javascript", css: "text/css" } as Record<string, string>)[
                    relative.split(".").pop()!
                ] ?? contentType(file);
            return new Response(readFileSync(file), {
                headers: { ...CROSS_ORIGIN_ISOLATION, ...(type && { "Content-Type": type }) },
            });
        },
    }),
);
beforeAll(async () => {
    browser = await publishTestChromium();
});
afterAll(async () => {
    await browser?.close();
    await Promise.all(servers.map((server) => server.stop(true)));
    await removeTestDirectory(root);
});

// minimal is the scaffold; physics loads its kernel chunk and workers; materials names an image by new URL(..., import.meta.url).
for (const example of ["minimal", "physics", "materials"])
    test(`a built ${example} loads at the root and under a subpath`, async () => {
        directory = join(root, example);
        await buildWeb(join(repo, "examples", example), { outDir: directory, stdout: 2 });
        const assets = publishAssets(directory);
        for (const [index, base] of ["/", subpath].entries()) {
            // The verifier fails on a console error, which every failed resource load logs.
            const verification = await verifyPublishedGame(
                browser,
                new URL(base, servers[index]!.url).href,
                directory,
                assets,
            );
            if (example === "physics") expect(verification.threads).toBeGreaterThan(1);
        }
    });
