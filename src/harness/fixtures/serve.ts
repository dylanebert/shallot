#!/usr/bin/env bun
// The fixture's serve command: bundles the fixture page for the browser and serves it. No dev server,
// watcher or cache — the driver owns the port and the process lifetime, and this process's output is the
// server evidence a failing row retains.

import { resolve } from "node:path";

const args = Bun.argv.slice(2);
const port = Number(args[args.indexOf("--port") + 1]);
if (!Number.isInteger(port) || port <= 0) {
    console.error("fixture serve refused: --port <number> is required");
    process.exit(1);
}

const built = await Bun.build({
    entrypoints: [resolve(import.meta.dir, "page.ts")],
    target: "browser",
});
if (!built.success) {
    console.error(`fixture serve refused: ${built.logs.map((log) => String(log)).join(" | ")}`);
    process.exit(1);
}
// `--fail` is the caller's own serve argument, so the deliberate-failure page is the same page with one
// declared flag rather than a second fixture.
const preamble = args.includes("--fail") ? "window.__fixtureFail = true;" : "";
const html = `<!doctype html><meta charset="utf-8"><title>capture fixture</title><script type="module">${preamble}${await built.outputs[0].text()}</script>`;

Bun.serve({
    port,
    fetch(request) {
        const path = new URL(request.url).pathname;
        // The browser asks for a favicon unprompted; answering it keeps the page-error diagnostics to
        // errors the page actually produced.
        if (path === "/favicon.ico") return new Response(null, { status: 204 });
        if (path !== "/") return new Response("not found", { status: 404 });
        return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    },
});
console.log(`fixture serve listening on ${port}`);
