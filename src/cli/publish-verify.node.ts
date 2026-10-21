import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";
import { CROSS_ORIGIN_ISOLATION } from "../project/isolation.js";
import { projectChromium, verifyPublishedGame } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const root = mkdtempSync(join(tmpdir(), "shallot-publish-controls-"));
const directory = join(root, "web");
mkdirSync(directory);
const script = 'dispatchEvent(new CustomEvent("shallot:startup",{detail:{kind:"app"}}));';
const html = `<link rel="icon" href="data:,"><script>${script}</script>`;
let control = "positive";
let browser: Awaited<ReturnType<typeof projectChromium>>;
const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
        const path = new URL(request.url).pathname;
        const headers: Record<string, string> = { ...CROSS_ORIGIN_ISOLATION, Connection: "close" };
        if (path === "/" && control === "absent isolation") {
            for (const name of Object.keys(CROSS_ORIGIN_ISOLATION)) delete headers[name];
        }
        if (path === "/game.js") {
            if (control === "missing asset")
                return new Response("missing", { status: 404, headers: { Connection: "close" } });
            return new Response(readFileSync(join(directory, "game.js")), {
                headers: {
                    ...headers,
                    "Content-Type":
                        control === "wrong MIME" ? "text/plain" : "application/javascript",
                },
            });
        }
        if (path === "/" || path === "/index.html" || control === "SPA fallback")
            return new Response(readFileSync(join(directory, "index.html")), {
                headers: { ...headers, "Content-Type": "text/html" },
            });
        return new Response("missing", { status: 404, headers: { Connection: "close" } });
    },
});
beforeAll(async () => {
    browser = await projectChromium(resolve(import.meta.dir, "../.."));
});
afterAll(async () => {
    await browser?.close();
    await server.stop(true);
    await removeTestDirectory(root);
});

for (const [fault, failure] of [
    ["positive", undefined],
    ["missing asset", "game.js: expected 200, got 404"],
    ["wrong MIME", "game.js: expected application/javascript, got text/plain"],
    ["SPA fallback", "A missing path answered 200, not 404"],
    ["absent isolation", "The published page is not cross-origin isolated"],
    ["no startup observation", `waitForFunction: Timeout ${TIMEOUT.startup}ms exceeded`],
] as const) {
    test(
        `the publish verifier ${failure ? `rejects ${fault}` : "accepts the isolated, observed positive control"}`,
        async () => {
            control = fault;
            writeFileSync(
                join(directory, "index.html"),
                fault === "no startup observation" ? '<link rel="icon" href="data:,">' : html,
            );
            writeFileSync(join(directory, "game.js"), "const game = 1;\n");
            const assets = ["index.html", "game.js"].map((path) => ({
                path,
                size: readFileSync(join(directory, path)).length,
            }));
            const result = verifyPublishedGame(browser, server.url.href, directory, assets);
            if (failure) await expect(result).rejects.toThrow(failure);
            else
                expect(await result).toMatchObject({ hardware: false, adapter: "No GPU required" });
        },
        // The missing-observation control exhausts the startup wait after loading the page.
        fault === "no startup observation" ? 2 * TIMEOUT.startup : TIMEOUT.node,
    );
}
