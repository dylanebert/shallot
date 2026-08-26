import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Routes this gate through `scripts/wsl-bridge.ts` when on WSL, so it drives the Windows host's real GPU
// instead of falling to the software adapter — the same shape `examples/showcase/roads/
// playwright.global-setup.ts` uses (that file's header has the full rationale: the endpoint-file handoff
// between this root process and each worker's fresh config re-import, and the known bridge-caller gaps).
// This gate is display-gated the same way: `test/touch.playwright.ts` skips on a software adapter.

const REPO_ROOT = resolve(import.meta.dirname, "../../..");
const BRIDGE_CONNECT = resolve(REPO_ROOT, "scripts/wsl-bridge-connect.ts");
export const ENDPOINT_FILE = resolve(
    import.meta.dirname,
    "node_modules/.cache/orbit-touch-bridge-endpoint.json",
);

const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

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
        handshake = await waitForHandshake(child, 150_000);
    } catch (e) {
        child.stdin?.end();
        console.log(
            `orbit-touch gate: bridge unavailable (${e instanceof Error ? e.message : e}), falling back to local adapter`,
        );
        return;
    }

    if ("skip" in handshake) {
        console.log(
            `orbit-touch gate: bridge unavailable (${handshake.skip}), falling back to local adapter`,
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
            `orbit-touch gate: bridge unavailable (couldn't write the endpoint handoff: ${e instanceof Error ? e.message : e}), falling back to local adapter`,
        );
        return;
    }

    return async () => {
        child.stdin?.end();
        await new Promise<void>((res) => child.on("exit", () => res()));
        rmSync(ENDPOINT_FILE, { force: true });
    };
}
