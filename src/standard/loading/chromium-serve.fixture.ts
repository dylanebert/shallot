import { resolve } from "node:path";

const args = Bun.argv.slice(2);
const port = Number(args[args.indexOf("--port") + 1]);
if (!Number.isInteger(port) || port <= 0) {
    console.error("loading chromium serve refused: --port <number> is required");
    process.exit(1);
}

const built = await Bun.build({
    entrypoints: [resolve(import.meta.dir, "chromium-page.fixture.ts")],
    target: "browser",
});
if (!built.success) {
    console.error(
        `loading chromium serve refused: ${built.logs.map((log) => String(log)).join(" | ")}`,
    );
    process.exit(1);
}
const html = `<!doctype html><meta charset="utf-8"><title>loading profile witness</title><script type="module">${await built.outputs[0].text()}</script>`;
Bun.serve({
    port,
    fetch(request) {
        if (new URL(request.url).pathname === "/favicon.ico")
            return new Response(null, { status: 204 });
        if (new URL(request.url).pathname !== "/")
            return new Response("not found", { status: 404 });
        return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    },
});
console.log(`loading chromium serve listening on ${port}`);
