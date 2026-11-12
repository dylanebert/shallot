import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { readStreams, referenceDirectory, WASM, WRAPPER } from "./reference";

const reference = referenceDirectory();
const streams = readStreams();
const built = await Bun.build({ entrypoints: [resolve(import.meta.dir, "browser.ts")], target: "browser" });
if (!built.success) throw new Error(String(built.logs));
const bundle = await built.outputs[0].text();
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const path = new URL(request.url).pathname;
    const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
    if (path === "/") return new Response('<script src="/reference.js"></script><script type="module" src="/browser.js"></script>', { headers: { ...headers, "Content-Type": "text/html" } });
    if (path === "/browser.js") return new Response(bundle, { headers: { ...headers, "Content-Type": "text/javascript" } });
    if (path === "/reference.js") return new Response(readFileSync(resolve(reference, WRAPPER)), { headers: { ...headers, "Content-Type": "text/javascript" } });
    if (path === "/reference.wasm") return new Response(readFileSync(resolve(reference, WASM)), { headers: { ...headers, "Content-Type": "application/wasm" } });
    if (path === "/streams") return Response.json(streams.map(({ bytes: _, ...stream }) => stream), { headers });
    const stream = path.startsWith("/stream/") ? streams[Number(path.slice(8))] : undefined;
    if (stream) return new Response(stream.bytes, { headers });
    return new Response("Not found", { status: 404, headers });
} });
let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
try {
    browser = await chromium.launch({ channel: "chrome", headless: true });
    console.log(`Chrome ${browser.version()}; reference ${WASM} at ${reference}`);
    for (let first = 0; first < 3; first++) {
        const page = await browser.newPage();
        page.on("console", (message) => { if (message.type() === "log") console.log(message.text()); });
        console.log(`Page ${first + 1}`);
        await page.goto(`http://127.0.0.1:${server.port}/#${first}`);
        await page.waitForFunction(() => {
            const state = globalThis as unknown as { failure?: string; done?: boolean };
            if (state.failure) throw new Error(state.failure);
            return state.done;
        }, undefined, { timeout: 0 });
        await page.close();
    }
} finally { await browser?.close(); server.stop(); }
