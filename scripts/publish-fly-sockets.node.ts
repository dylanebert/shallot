import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { BROWSER_CONFIG, TIMEOUT } from "@dylanebert/shallot/testing";
import { chromium } from "playwright";
import {
    assertFlySocketsClosed,
    type FlySocketObservation,
    observeFlyClients,
} from "./publish-fly-sockets";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

test("idle refuses every unclosed socket or browser and rejects unmatched closes", () => {
    const rows: FlySocketObservation[] = [
        { kind: "browser-created", id: "browser:1", at: 1 },
        { kind: "socket-created", id: "browser:socket:1", at: 2 },
        { kind: "socket-first-frame", id: "browser:socket:1", at: 2 },
    ];
    expect(() => assertFlySocketsClosed(rows)).toThrow("browser:1, browser:socket:1");
    rows.push({ kind: "socket-closed", id: "browser:socket:1", at: 3 });
    expect(() => assertFlySocketsClosed(rows)).toThrow("browser:1");
    rows.push({ kind: "browser-closed", id: "browser:1", at: 4 });
    expect(() => assertFlySocketsClosed(rows)).not.toThrow();
    expect(() => assertFlySocketsClosed([{ kind: "socket-closed", id: "unknown", at: 5 }])).toThrow(
        "Close without creation",
    );
});

test("Bun clients and a Node-launched browser record actual socket closes before idle", async () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-fly-sockets-"));
    const observations: FlySocketObservation[] = [];
    const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch(request, server) {
            if (server.upgrade(request)) return;
            return new Response(
                `<script>window.socket=new WebSocket('ws://${server.hostname}:${server.port}/socket')</script>`,
                { headers: { "Content-Type": "text/html" } },
            );
        },
        websocket: {
            open(socket) {
                socket.send("first");
                socket.send("second");
            },
            message() {},
        },
    });
    const restoreClients = observeFlyClients((row) => observations.push(row), "hot");
    try {
        const socket = new WebSocket(`ws://${server.hostname}:${server.port}/socket`);
        await new Promise<void>((accept, reject) => {
            socket.addEventListener("message", () => accept(), { once: true });
            socket.addEventListener("error", reject, { once: true });
        });
        expect(() => assertFlySocketsClosed(observations)).toThrow("bun:hot:socket:1");
        const closed = new Promise<void>((accept) =>
            socket.addEventListener("close", () => accept(), { once: true }),
        );
        socket.close();
        await closed;
        restoreClients();
        assertFlySocketsClosed(observations);
        const entry = join(root, "browser.ts");
        writeFileSync(
            entry,
            `import {createRequire} from 'node:module';
import {observeFlyBrowser,assertFlySocketsClosed} from ${JSON.stringify(join(import.meta.dir, "publish-fly-sockets.ts"))};
const projectRequire=createRequire(${JSON.stringify(resolve(import.meta.dir, "../package.json"))});
const {chromium}=projectRequire('playwright'); const observations=[];
observeFlyBrowser(chromium,row=>observations.push(row));
(async()=>{const browser=await chromium.launch(${JSON.stringify({ ...BROWSER_CONFIG.use.launchOptions, channel: BROWSER_CONFIG.use.channel, executablePath: chromium.executablePath() })});
try{const context=await browser.newContext();const page=await context.newPage();await page.goto(${JSON.stringify(server.url.href)});await page.waitForFunction(()=>window.socket.readyState===WebSocket.OPEN);await context.close();}
finally{await browser.close();}
assertFlySocketsClosed(observations);console.log(JSON.stringify(observations));})().catch(error=>{console.error(error);process.exitCode=1;});`,
        );
        const build = await Bun.build({
            entrypoints: [entry],
            target: "node",
            format: "cjs",
            outdir: root,
            naming: "browser.cjs",
        });
        expect(build.success, build.logs.join("\n")).toBe(true);
        const child = Bun.spawn(["node", join(root, "browser.cjs")], {
            stdout: "pipe",
            stderr: "pipe",
            timeout: TIMEOUT.node,
        });
        const [code, output, error] = await Promise.all([
            child.exited,
            new Response(child.stdout).text(),
            new Response(child.stderr).text(),
        ]);
        expect(code, error).toBe(0);
        observations.push(...JSON.parse(output));
        assertFlySocketsClosed(observations);
        expect(observations.map((row) => [row.kind, row.id])).toEqual([
            ["socket-created", "bun:hot:socket:1"],
            ["socket-first-frame", "bun:hot:socket:1"],
            ["socket-closed", "bun:hot:socket:1"],
            ["browser-created", "browser:1"],
            ["socket-created", "browser:socket:1"],
            ["socket-first-frame", "browser:socket:1"],
            ["socket-closed", "browser:socket:1"],
            ["browser-closed", "browser:1"],
        ]);
        expect(observations.every((row) => Number.isFinite(row.at))).toBe(true);
    } finally {
        restoreClients();
        await server.stop(true);
        await removeTestDirectory(root);
    }
});
