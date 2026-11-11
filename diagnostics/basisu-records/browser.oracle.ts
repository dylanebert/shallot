import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { bunChild } from "../../scripts/bun-child";
import type { Input } from "./comparison";
import { controls, readInputs } from "./controls";
import { referenceDirectory } from "./reference";

const root = resolve(import.meta.dir, "../..");
const reference = referenceDirectory();
const sync = Bun.spawnSync({ ...bunChild([resolve(import.meta.dir, "records.ts")]), cwd: root, stdout: "pipe", stderr: "pipe" });
if (sync.exitCode !== 0) throw new Error(new TextDecoder().decode(sync.stderr));
const files: Input[] = process.argv.includes("--attribution") ? controls() : readInputs();
const built = await Bun.build({ entrypoints: [resolve(import.meta.dir, "browser.ts")], target: "browser" });
if (!built.success) throw new Error(String(built.logs));
const bundle = await built.outputs[0].text();
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const path = decodeURIComponent(new URL(request.url).pathname);
    const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
    if (path === "/") return new Response('<script src="/reference.js"></script><script type="module" src="/browser.js"></script>', { headers: { ...headers, "Content-Type": "text/html" } });
    if (path === "/browser.js") return new Response(bundle, { headers: { ...headers, "Content-Type": "text/javascript" } });
    if (path === "/reference.js") return new Response(readFileSync(resolve(reference, "webgl/transcoder/build/basis_transcoder.js")), { headers: { ...headers, "Content-Type": "text/javascript" } });
    if (path === "/reference.wasm") return new Response(readFileSync(resolve(reference, "webgl/transcoder/build/basis_transcoder.wasm")), { headers });
    if (path === "/inputs") return Response.json(files.map((file) => file.path), { headers });
    for (const [prefix, field] of [["/input/", "bytes"], ["/plain/", "plain"]] as const) {
        if (path.startsWith(prefix)) {
            const file = files.find((file) => file.path === path.slice(prefix.length));
            if (file?.[field]) return new Response(file[field], { headers });
        }
    }
    return new Response("Not found", { status: 404, headers });
} });
let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
try {
    browser = await chromium.launch({ channel: "chrome", headless: true });
    console.log(`Chrome ${browser.version()}; pinned reference ${reference}`);
    for (let first = 0; first < 3; first++) {
        const page = await browser.newPage();
        page.on("console", (message) => { if (message.type() === "log") console.log(message.text()); });
        page.on("pageerror", (error) => { throw error; });
        const profiler = process.argv.includes("--profile") ? await page.context().newCDPSession(page) : undefined;
        if (profiler) {
            await profiler.send("Profiler.enable");
            await profiler.send("Profiler.start");
        }
        console.log(`Page ${first + 1}`);
        await page.goto(`http://127.0.0.1:${server.port}/${process.argv.includes("--attribution") ? "?attribution" : ""}#${first}`);
        await page.waitForFunction(() => {
            const state = globalThis as unknown as { failure?: string; done?: boolean };
            if (state.failure) throw new Error(state.failure);
            return state.done;
        }, undefined, { timeout: 0 });
        if (profiler) {
            const { profile } = await profiler.send("Profiler.stop");
            await Bun.write(resolve(root, `test-results/basisu-chrome-profile-${first + 1}.json`), JSON.stringify(profile));
        }
        await page.close();
    }
} finally { await browser?.close(); server.stop(); }
