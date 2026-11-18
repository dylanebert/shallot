import { expect, setDefaultTimeout, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "../../scripts/bun-child";
import {
    createExample,
    example,
    packedInstall,
    usePackedInstall,
} from "../../scripts/packed-install-fixture";

setDefaultTimeout(TIMEOUT.node);
usePackedInstall();

test("Node builds a packed multiplayer server bundle serving two Worlds that each confirm 60 ticks via PORT", async () => {
    createExample("multiplayer");
    const { project } = example("multiplayer");
    const directory = join(packedInstall().scratch, "server");
    mkdirSync(directory);
    const module = join(directory, "publish-fly.mjs");
    const build = await Bun.build({
        entrypoints: [join(import.meta.dir, "publish-fly.ts")],
        target: "node",
        outdir: directory,
        naming: "publish-fly.mjs",
    });
    expect(build.success, build.logs.join("\n")).toBe(true);
    const published = spawnSync(
        "node",
        [
            "--input-type=module",
            "-e",
            `
            import { buildFlyServer } from ${JSON.stringify(pathToFileURL(module).href)};
            await buildFlyServer(${JSON.stringify(project)}, ${JSON.stringify(directory)}, "game");
        `,
        ],
        { cwd: project, encoding: "utf8", timeout: TIMEOUT.node },
    );
    expect(published.status, published.stderr).toBe(0);
    const version = spawnSync("bun", ["--version"], { cwd: project, encoding: "utf8" });
    expect(version.status, version.stderr).toBe(0);
    expect(readFileSync(join(directory, "Dockerfile"), "utf8")).toBe(
        `FROM oven/bun:${version.stdout.trim()}\nWORKDIR /app\nCOPY server.js .\nCMD ["bun", "server.js"]\n`,
    );
    const runner = join(project, "confirm.ts");
    await Bun.write(
        runner,
        `
import { createApp } from "@dylanebert/shallot";
import { client } from "@dylanebert/shallot/multiplayer";
import { ClientSession, GAME } from "./src/network";
const apps = [];
let error;
try {
    for (let i = 0; i < 2; i++) apps.push(await createApp({ defaults: false,
        plugins: [client(GAME, { url: process.argv[2], onError: value => { error = value; } })] }));
    const confirmed = () => apps.map(app => app.world.resource(ClientSession).client?.prediction.receiver.confirmed ?? 0);
    let previous = performance.now();
    while (confirmed().some(tick => tick < 60)) {
        await Bun.sleep(5);
        if (error) throw error;
        const now = performance.now();
        for (const app of apps) app.world.step((now - previous) / 1000);
        previous = now;
    }
    console.log(JSON.stringify(confirmed()));
} finally { for (const app of apps) app.dispose(); }
`,
    );
    const server = Bun.spawn({
        ...bunChild([join(directory, "server.js")]),
        cwd: directory,
        env: { ...process.env, PORT: "0" },
        stdout: "pipe",
        stderr: "inherit",
        timeout: TIMEOUT.node,
    });
    let clients: Bun.Subprocess<"ignore", "pipe", "pipe"> | undefined;
    try {
        const reader = server.stdout.getReader();
        let output = "";
        let url: string | undefined;
        while (!url) {
            const chunk = await reader.read();
            if (chunk.done) throw new Error(`Server exited without an address: ${output}`);
            output += new TextDecoder().decode(chunk.value);
            url = output.match(/ws:\/\/localhost:\d+/)?.[0];
        }
        reader.releaseLock();
        expect(new URL(url).port).not.toBe("3001");
        expect(new URL(url).port).not.toBe("0");
        clients = Bun.spawn({
            ...bunChild([runner, url]),
            cwd: project,
            stdin: "ignore",
            stdout: "pipe",
            stderr: "pipe",
            timeout: TIMEOUT.node,
        });
        const result = await new Response(clients.stdout).text();
        const errors = await new Response(clients.stderr).text();
        expect(await clients.exited, errors).toBe(0);
        const ticks = JSON.parse(result.trim());
        expect(ticks).toHaveLength(2);
        for (const tick of ticks) expect(tick).toBeGreaterThanOrEqual(60);
    } finally {
        clients?.kill();
        if (clients) await clients.exited;
        server.kill();
        await server.exited;
    }
});
