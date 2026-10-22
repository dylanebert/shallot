import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, statSync, symlinkSync, writeFileSync } from "node:fs";
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
let blockWorkers = false;
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
    const newContext = browser.newContext.bind(browser);
    browser.newContext = async (...args) => {
        const context = await newContext(...args);
        await context.addInitScript((blocked) => {
            Object.defineProperty(navigator, "hardwareConcurrency", { value: blocked ? 4 : 3 });
            if (blocked)
                window.Worker = class {
                    constructor() {
                        throw new Error("Worker startup blocked by test");
                    }
                } as unknown as typeof Worker;
        }, blockWorkers);
        return context;
    };
});
afterAll(async () => {
    await browser?.close();
    await Promise.all(servers.map((server) => server.stop(true)));
    await removeTestDirectory(root);
});

test("the verifier rejects physics after worker startup falls back to the single artifact", async () => {
    const project = join(root, "physics-fallback");
    mkdirSync(project);
    symlinkSync(join(repo, "node_modules"), join(project, "node_modules"), "junction");
    writeFileSync(
        join(project, "vite.config.mts"),
        `export default { resolve: { alias: {
            "fixture-app": ${JSON.stringify(join(repo, "src/engine/app/index.ts"))},
            "fixture-physics": ${JSON.stringify(join(repo, "src/standard/physics/runtime.ts"))}
        } } };`,
    );
    writeFileSync(
        join(project, "index.html"),
        `<link rel="icon" href="data:,"><script type="module">
            import { runApp } from "fixture-app";
            import { StandardPhysicsPlugin } from "fixture-physics";
            await runApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        </script>`,
    );
    directory = join(project, "dist");
    await buildWeb(project, { stdout: 2 });
    blockWorkers = true;
    try {
        await expect(
            verifyPublishedGame(browser, servers[0]!.url.href, directory, publishAssets(directory)),
        ).rejects.toThrow("Physics initialized the single artifact, not shared.");
    } finally {
        blockWorkers = false;
    }
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
            if (example === "physics") {
                expect(verification.artifact).toBe("shared");
                expect(verification.threads).toBe(1);
            }
        }
    });
