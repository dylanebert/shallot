import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Socket } from "node:net";
import { join, resolve } from "node:path";

const REPO_ROOT = resolve(import.meta.dir, "..");

// The WSL → Windows real-GPU bridge for `shallot verify`. WSL has no conformant WebGPU adapter, so the gates
// would skip; this drives the Windows host's real-GPU Chrome instead, keeping verify's own server + drive
// logic on the WSL side. It exists entirely in `scripts/` (repo tooling) — none of this ships in
// @dylanebert/shallot, whose only remote-browser surface is the generic `shallot verify --connect <ws>`.
//
// Three facts about this WSL+Windows setup force the shape below (each validated empirically, not assumed):
//
//  1. WSL → host TCP is blocked (Hyper-V firewall, DefaultInboundAction=Block, non-admin can't open it);
//     host → WSL works (localhost forwarding + the WSL eth0 IP). So the browser server on the host can't be
//     dialed from WSL directly. A host-initiated reverse tunnel carries the one Playwright ws connection:
//     the host dials a WSL rendezvous (allowed direction), and each accepted stream is spliced to the
//     host-local browser ws. verify then connects to a WSL-local port that the rendezvous fronts.
//
//  2. Bun's Playwright client hangs after the ws upgrade on this platform (`chromium.connect` never
//     resolves, `chromium.launch` never returns — reproduced with no tunnel at all). Node's client works.
//     verify only runs under Bun (bun-style extensionless imports across its toolchain graph), so we bundle
//     the verify CLI to a node-runnable file and drive it with `node` on the WSL side.
//
//  3. The Playwright connect protocol requires the client and server versions to match exactly (a mismatch
//     answers the upgrade with `428 Precondition Required`). The host is pinned to the same version this
//     repo's Playwright resolves, read programmatically, and re-provisioned when it drifts.
//
// A prerequisite the bridge can't satisfy (no powershell interop, no node/bun on the host) is an honest skip
// with a reason; anything past those prerequisites either works or fails loud — never a hang, never a pass.
//
// Coverage exercised over --connect and green: recipes, flows, and `bun bench` (all the same verify path).
// `--memory` samples over --connect too — the retained-leak slope is a real reading across the bridge: the
// forced-GC + getMetrics CDP RPCs (`HeapProfiler.*` / `Performance.*`) are Playwright-protocol-generic, so
// the tunnel forwards them untouched. A run too short for three 800ms samples reports null; an old note here
// read that degradation as "returns null over --connect", which was never a code guarantee. Untested arms:
// mirrored-networking WSL, and `--alloc` over --connect.

const STAGE_NAME = "shallot-verify-bridge";
const BUNDLE = resolve(REPO_ROOT, "node_modules/.cache/shallot-wsl-verify.mjs");
const BIN_DIR = resolve(REPO_ROOT, "packages/shallot/bin");
const POOL = 8;

function sh(cmd: string): { ok: boolean; out: string } {
    const p = Bun.spawnSync(["powershell.exe", "-NoProfile", "-Command", cmd], {
        stdout: "pipe",
        stderr: "pipe",
    });
    return {
        ok: p.exitCode === 0,
        out: new TextDecoder().decode(p.stdout).trim().replace(/\r/g, ""),
    };
}

/** the pinned Playwright version — the one this repo's client uses, which the host must match exactly. */
function playwrightVersion(): string {
    return JSON.parse(
        readFileSync(resolve(REPO_ROOT, "node_modules/playwright/package.json"), "utf8"),
    ).version;
}

/** the WSL eth0 address the host dials back on (host → WSL is the reachable direction). */
function wslHostIp(): string {
    const p = Bun.spawnSync(["ip", "-4", "-o", "addr", "show", "eth0"], { stdout: "pipe" });
    const m = new TextDecoder().decode(p.stdout).match(/inet (\d+\.\d+\.\d+\.\d+)/);
    if (!m) throw new Error("could not read the WSL eth0 IP (ip addr show eth0)");
    return m[1];
}

/** an OS-picked free TCP port on the WSL side (bind :0, read it, release). */
function freePort(host = "127.0.0.1"): Promise<number> {
    return new Promise((res, rej) => {
        const s = createServer();
        s.on("error", rej);
        s.listen(0, host, () => {
            const p = (s.address() as { port: number }).port;
            s.close(() => res(p));
        });
    });
}

// The host launcher: launches the real-GPU browser server (the exact channel + WebGPU flags the shipped gate
// launches with, so the connected browser is the same browser), then opens a pool of reverse tunnels. Each
// dials the WSL rendezvous; the first bytes on a tunnel (verify's ws upgrade) trigger a dial to the local
// browser ws, and the two are spliced. Runs under node (fact 2). Its pid goes to a file so the WSL side can
// tree-kill it (the reliable teardown); the rendezvous watchdog below is the self-terminating backstop.
const LAUNCHER = `
import net from "node:net";
import { writeFileSync } from "node:fs";
const RV_HOST = process.env.RV_HOST, RV_PORT = Number(process.env.RV_PORT), POOL = Number(process.env.POOL || 8);
try {
  const { chromium } = await import("playwright");
  const server = await chromium.launchServer({
    headless: true,
    channel: "chromium",
    args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures"],
  });
  const ep = new URL(server.wsEndpoint());
  const wsPort = Number(ep.port), wsHost = ep.hostname;
  writeFileSync(process.env.PID_FILE, String(process.pid));
  writeFileSync(process.env.ENDPOINT_FILE, ep.pathname.slice(1));
  // Liveness rides the pool, not stdin: the pipe here is bun -> powershell -> node, so node never sees the
  // WSL side's EOF. Instead the pooled tunnels stay connected to the WSL rendezvous while the bridge lives;
  // when it tears down they all close and reconnects fail, so "no open tunnel after we'd had some" is the
  // exit signal. The browser server's own handles would otherwise keep node alive forever.
  const live = new Set();
  let everOpen = false;
  function tunnel() {
    const rv = net.connect(RV_PORT, RV_HOST);
    let browser = null, buffered = [];
    rv.on("connect", () => { live.add(rv); everOpen = true; });
    rv.on("error", () => rv.destroy());
    rv.on("data", (d) => {
      if (!browser) {
        browser = net.connect(wsPort, wsHost, () => { for (const b of buffered) browser.write(b); buffered = null; });
        browser.on("data", (x) => rv.write(x));
        browser.on("error", () => rv.destroy());
        browser.on("close", () => rv.destroy());
        tunnel(); // replenish the consumed slot
      }
      if (buffered) buffered.push(d); else browser.write(d);
    });
    rv.on("close", () => {
      live.delete(rv);
      if (browser) browser.destroy();
      else setTimeout(tunnel, 200); // an idle tunnel dropped — keep the pool full while the bridge lives
    });
  }
  for (let i = 0; i < POOL; i++) tunnel();
  let starved = 0;
  setInterval(() => {
    starved = live.size === 0 ? starved + 1 : 0;
    if (everOpen && starved >= 4) { server.close().catch(() => {}); setTimeout(() => process.exit(0), 200); }
  }, 500).unref();
} catch (e) { writeFileSync(process.env.ENDPOINT_FILE, "ERROR " + (e && e.stack || e)); process.exit(1); }
`;

interface Staged {
    win: string;
    wsl: string;
}

/** ensure a host staging dir carrying the launcher + a Playwright pinned to `version`, installed. Reinstalls
 *  only when the host's installed version drifts from the pin (the connect protocol needs an exact match). */
function provisionHost(version: string): Staged {
    const temp = sh("Write-Host -NoNewline $env:TEMP");
    if (!temp.ok || !temp.out) throw new Error("no Windows TEMP (powershell interop)");
    const win = `${temp.out}\\${STAGE_NAME}`;
    const wslTemp = new TextDecoder()
        .decode(Bun.spawnSync(["wslpath", temp.out], { stdout: "pipe" }).stdout)
        .trim();
    const wsl = join(wslTemp, STAGE_NAME);

    mkdirSync(wsl, { recursive: true });
    writeFileSync(
        join(wsl, "package.json"),
        `${JSON.stringify({ name: STAGE_NAME, private: true, dependencies: { playwright: version } }, null, 2)}\n`,
    );
    writeFileSync(join(wsl, "launch.mjs"), LAUNCHER);

    const installed = existsSync(join(wsl, "node_modules/playwright/package.json"))
        ? JSON.parse(readFileSync(join(wsl, "node_modules/playwright/package.json"), "utf8"))
              .version
        : null;
    if (installed !== version) {
        const r = sh(`cd '${win}'; bun install --silent; bunx playwright install chromium`);
        if (!r.ok)
            throw new Error(
                `host provisioning failed (bun install / playwright install):\n${r.out}`,
            );
    }
    return { win, wsl };
}

/** build the verify CLI to a node-runnable bundle. Bun's `import.meta.dir` (used across bin/) is defined to
 *  the bin dir — the only place it appears in the graph — and the toolchain's native deps stay external so
 *  node resolves them from this repo's node_modules. Rebuilt every start: the build is fast and drift-free. */
function buildBundle(): void {
    mkdirSync(resolve(REPO_ROOT, "node_modules/.cache"), { recursive: true });
    const r = Bun.spawnSync(
        [
            "bun",
            "build",
            "packages/shallot/bin/cli.ts",
            "--target",
            "node",
            "--outfile",
            BUNDLE,
            "--define",
            `import.meta.dir="${BIN_DIR}"`,
            "--external",
            "vite",
            "--external",
            "playwright",
            "--external",
            "lightningcss",
            "--external",
            "@swc/*",
            "--external",
            "esbuild",
            "--external",
            "rollup",
            "--external",
            "fsevents",
        ],
        { cwd: REPO_ROOT, stdout: "pipe", stderr: "pipe" },
    );
    if (r.exitCode !== 0) {
        throw new Error(`node bundle build failed:\n${new TextDecoder().decode(r.stderr)}`);
    }
}

export interface Bridge {
    /** ws endpoint to pass verify as `--connect` (a WSL-local port the reverse tunnel fronts). */
    connectUrl: string;
    /** the node-runnable verify bundle to spawn with `node`. */
    bundle: string;
    teardown(): Promise<void>;
}

/** the WSL bridge's non-negotiable prerequisites, probed cheaply. A missing one is a legitimate skip; when
 *  all pass, `start()` either works or fails loud. */
export function bridgePrereq(): string | null {
    const both = sh("node --version; bun --version");
    if (!both.ok)
        return "no Windows-host node/bun (or no powershell interop) for the real-GPU bridge";
    const [node, bun] = both.out.split("\n");
    if (!/^v?\d/.test(node ?? ""))
        return "no node on the Windows host (the bridge launcher runs under node)";
    if (!/^\d/.test(bun ?? ""))
        return "no bun on the Windows host (needed to install the bridge's Playwright)";
    return null;
}

/** provision + launch the bridge: host browser server, the reverse tunnel, and the node verify bundle.
 *  Resolves a handle, or throws with a reason naming what failed (the caller surfaces it and fails loud). */
export async function start(): Promise<Bridge> {
    const version = playwrightVersion();
    const rvHost = wslHostIp();
    const staged = provisionHost(version);
    buildBundle();

    const rvPort = await freePort("0.0.0.0");
    const clientPort = await freePort("127.0.0.1");

    // rendezvous: the host dials in; each accepted stream waits in a pool for a verify client to pair with.
    const pool: Socket[] = [];
    const waiters: ((s: Socket) => void)[] = [];
    const rv = createServer((sock) => {
        sock.on("error", () => {});
        const w = waiters.shift();
        if (w) w(sock);
        else pool.push(sock);
    });
    const takeTunnel = (): Promise<Socket> => {
        const s = pool.shift();
        return s ? Promise.resolve(s) : new Promise((res) => waiters.push(res));
    };
    // client: verify connects here; splice it to a pooled host tunnel (raw bytes — the ws rides through).
    const client = createServer(async (c) => {
        c.on("error", () => {});
        const t = await takeTunnel();
        c.pipe(t);
        t.pipe(c);
        t.on("close", () => c.destroy());
        c.on("close", () => t.destroy());
    });
    await new Promise<void>((res, rej) => {
        rv.on("error", rej);
        rv.listen(rvPort, "0.0.0.0", res);
    });
    await new Promise<void>((res, rej) => {
        client.on("error", rej);
        client.listen(clientPort, "127.0.0.1", res);
    });

    const endpointFile = join(staged.wsl, "endpoint.txt");
    const pidFile = join(staged.wsl, "pid.txt");
    rmSync(endpointFile, { force: true });
    rmSync(pidFile, { force: true });
    const launcher = Bun.spawn(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            `$env:ENDPOINT_FILE='${staged.win}\\endpoint.txt'; $env:PID_FILE='${staged.win}\\pid.txt'; $env:RV_HOST='${rvHost}'; $env:RV_PORT='${rvPort}'; $env:POOL='${POOL}'; cd '${staged.win}'; node launch.mjs`,
        ],
        { stdin: "pipe", stdout: "inherit", stderr: "inherit" },
    );

    // The launcher is a grandchild (bun → powershell → node), so its stdin isn't ours to EOF and killing the
    // powershell leaves node reparented. `taskkill /T` on the launcher's own pid (it writes it to pid.txt)
    // ends node + its browser children as a tree — the deterministic teardown, synchronous so it also works
    // from a process-exit hook. The launcher's own rendezvous watchdog is the backstop if this never runs.
    const killTree = (): void => {
        rv.close();
        client.close();
        for (const s of pool) s.destroy();
        const pid = existsSync(pidFile) ? readFileSync(pidFile, "utf8").trim() : "";
        if (/^\d+$/.test(pid))
            Bun.spawnSync(["taskkill.exe", "/PID", pid, "/T", "/F"], {
                stdout: "ignore",
                stderr: "ignore",
            });
        try {
            launcher.stdin.end();
        } catch {}
    };
    const teardown = async (): Promise<void> => {
        killTree();
        await launcher.exited.catch(() => {});
    };

    // wait for the launcher to publish the browser ws guid (or report an error / hang past the budget)
    const deadline = Date.now() + 90_000;
    let guid = "";
    while (Date.now() < deadline) {
        if (existsSync(endpointFile)) {
            guid = readFileSync(endpointFile, "utf8").trim();
            if (guid) break;
        }
        await Bun.sleep(300);
    }
    if (!guid || guid.startsWith("ERROR")) {
        await teardown();
        throw new Error(
            guid
                ? guid.replace(/^ERROR\s*/, "host launcher: ")
                : "host browser server never came up (launcher timed out)",
        );
    }

    // a process-exit hook can only run sync work, so the tree kill (spawnSync) is the reliable teardown when
    // the driver ends via process.exit — the async `teardown()` is for explicit callers.
    process.on("exit", killTree);
    return { connectUrl: `ws://localhost:${clientPort}/${guid}`, bundle: BUNDLE, teardown };
}
