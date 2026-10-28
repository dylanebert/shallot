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
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { publishTestChromium } from "../../scripts/publish-browser";
import { removeTestDirectory } from "../../scripts/test-directory";
import { buildWeb, CROSS_ORIGIN_ISOLATION, contentType } from "../project";
import { verifyPublishedGame } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);
const repo = resolve(import.meta.dir, "../..");
const root = mkdtempSync(join(tmpdir(), "shallot-publish-fallback-"));
const project = join(root, "physics-fallback");
const directory = join(project, "dist");
const TYPES: Record<string, string> = { html: "text/html", js: "text/javascript", css: "text/css" };
const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
        const relative = new URL(request.url).pathname.slice(1) || "index.html";
        const file = join(directory, relative);
        if (relative.includes("..") || !statSync(file, { throwIfNoEntry: false })?.isFile())
            return new Response("missing", { status: 404 });
        const type = TYPES[relative.split(".").pop()!] ?? contentType(file);
        return new Response(readFileSync(file), {
            headers: { ...CROSS_ORIGIN_ISOLATION, ...(type && { "Content-Type": type }) },
        });
    },
});
let browser: Awaited<ReturnType<typeof publishTestChromium>>;
beforeAll(async () => {
    browser = await publishTestChromium();
    const newContext = browser.newContext.bind(browser);
    browser.newContext = async (...args) => {
        const context = await newContext(...args);
        await context.addInitScript(() => {
            Object.defineProperty(navigator, "hardwareConcurrency", { value: 4 });
            window.Worker = class {
                constructor() {
                    throw new Error("Worker startup blocked by test");
                }
            } as unknown as typeof Worker;
        });
        return context;
    };
});
afterAll(async () => {
    await browser?.close();
    await server.stop(true);
    await removeTestDirectory(root);
});

test("the verifier rejects physics after worker startup falls back to the single artifact, before app startup", async () => {
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
            import { createApp } from "fixture-app";
            import { StandardPhysicsPlugin } from "fixture-physics";
            await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        </script>`,
    );
    await buildWeb(project, { stdout: 2 });
    expect(
        await verifyPublishedGame(browser, server.url.href, directory, []).then(
            () => undefined,
            (error: Error) => error.message,
        ),
    ).toContain("Physics initialized the single artifact, not shared.");
});
