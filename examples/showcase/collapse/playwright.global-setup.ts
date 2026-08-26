import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Routes this project's gate through `scripts/wsl-bridge.ts` when on WSL, so it drives the Windows host's
// real GPU instead of falling to the software adapter (the driver's own adapter-name skip). Identical
// shape to `examples/showcase/roads/playwright.global-setup.ts` — see that file's header for the three
// platform facts (blocked WSL→host TCP, bun's Playwright client hanging on connect, exact-version match)
// that force it. `scripts/wsl-bridge-connect.ts` is the one bun subprocess this process spawns and holds
// open for the run's duration.
//
// `playwright.config.ts` can't read the discovered endpoint directly — it's resolved *after* config load,
// and a worker process re-imports the config fresh rather than sharing this process's memory. The endpoint
// file is the handoff Playwright's own docs name for this shape: globalSetup runs once in the root process
// before any worker spawns, writes what it found, and each worker's later re-import reads it back.
//
// Roads' known, accepted gaps apply here unchanged: this is another caller into `wsl-bridge.ts`'s
// single-session bridge, so running this gate concurrently with `bun bench`/`bun run flows` races the same
// host staging dir; and the bridge-launched browser ignores `playwright.config.ts`'s
// `--enable-dawn-features=allow_unsafe_apis` (`connectOptions` makes `launchOptions` inert) — unused by
// collapse's gates today, so no live failure, but a future Dawn-unsafe-API use would only hit it over the
// bridge path.
const REPO_ROOT = resolve(import.meta.dirname, "../../..");
const BRIDGE_CONNECT = resolve(REPO_ROOT, "scripts/wsl-bridge-connect.ts");
export const ENDPOINT_FILE = resolve(
    import.meta.dirname,
    "node_modules/.cache/collapse-bridge-endpoint.json",
);

const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

/** scan stdout lines for the connect script's one handshake line — never assume first or last, since the
 *  host launcher's own inherited output can land on either side of it (mirrors `scripts/verify.ts`'s
 *  `extractResult` scan-don't-assume idiom). */
function waitForHandshake(
    child: ReturnType<typeof spawn>,
    timeoutMs: number,
): Promise<{ skip: string } | { connectUrl: string }> {
    return new Promise((resolve, reject) => {
        let buf = "";
        const timer = setTimeout(
            () => reject(new Error("bridge connect: no handshake within timeout")),
            timeoutMs,
        );
        const onData = (chunk: Buffer) => {
            buf += chunk.toString("utf8");
            for (const line of buf.split("\n")) {
                const m = line.match(/^BRIDGE_ENDPOINT (.+)$/);
                if (m) {
                    clearTimeout(timer);
                    child.stdout?.off("data", onData);
                    try {
                        resolve(JSON.parse(m[1]));
                    } catch (e) {
                        reject(new Error(`bridge connect: malformed handshake payload: ${e}`));
                    }
                    return;
                }
            }
        };
        child.stdout?.on("data", onData);
        child.on("error", (e) => {
            clearTimeout(timer);
            reject(e);
        });
        child.on("exit", (code) => {
            if (code !== 0) {
                clearTimeout(timer);
                reject(new Error(`bridge connect exited ${code} before a handshake`));
            }
        });
    });
}

export default async function globalSetup(): Promise<(() => Promise<void>) | void> {
    rmSync(ENDPOINT_FILE, { force: true });
    if (!isWSL) return;

    const child = spawn("bun", [BRIDGE_CONNECT], {
        cwd: REPO_ROOT,
        stdio: ["pipe", "pipe", "inherit"],
    });

    let handshake: { skip: string } | { connectUrl: string };
    try {
        // provisioning (host `bun install` + `playwright install chromium` on drift) plus the launcher's own
        // 90 s deadline (`wsl-bridge.ts`) — generous enough for a cold host cache, still bounded.
        handshake = await waitForHandshake(child, 150_000);
    } catch (e) {
        child.stdin?.end();
        console.log(
            `collapse gate: bridge unavailable (${e instanceof Error ? e.message : e}), falling back to local adapter`,
        );
        return;
    }

    if ("skip" in handshake) {
        console.log(
            `collapse gate: bridge unavailable (${handshake.skip}), falling back to local adapter`,
        );
        child.stdin?.end();
        return;
    }

    try {
        mkdirSync(resolve(ENDPOINT_FILE, ".."), { recursive: true });
        writeFileSync(ENDPOINT_FILE, JSON.stringify({ wsEndpoint: handshake.connectUrl }));
    } catch (e) {
        child.stdin?.end();
        console.log(
            `collapse gate: bridge unavailable (couldn't write the endpoint handoff: ${e instanceof Error ? e.message : e}), falling back to local adapter`,
        );
        return;
    }

    return async () => {
        child.stdin?.end();
        await new Promise<void>((res) => child.on("exit", () => res()));
        rmSync(ENDPOINT_FILE, { force: true });
    };
}
