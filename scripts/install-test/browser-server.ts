import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { REAL_GPU_LAUNCH } from "@dylanebert/shallot/harness/browser";

// The frozen previous release's CLI hard-codes a headless launch, so these flows stand up a matching
// full-Chromium browser server and attach to it. The server uses the previous install's own Playwright;
// shared by the compatibility and output flows, which both run a `previous` label.

/** the launch options the previous release's browser server takes: the published full-Chromium recipe,
 *  headless, matching its historical `verify` mode while retaining real hardware through the channel. */
export const CONNECT_LAUNCH = { ...REAL_GPU_LAUNCH, headless: true };

/** the `shallot verify` argv for one label. Only `previous` attaches to a browser server: its CLI
 *  predates the headed default and can reach real hardware no other way, while `candidate` must keep
 *  exercising its own launch — that launch is part of what this flow verifies. `dir` is the project to
 *  verify, which the output flow varies and the compatibility flow leaves at the app itself. */
export function verifyArgs(label: "previous" | "candidate", endpoint: string, dir = "."): string[] {
    const argv = ["verify", dir, "--dist", "--json", "--timeout", "30000"];
    return label === "previous" ? [...argv, "--connect", endpoint] : argv;
}

/** the ws endpoint a `launchServer` child announces on stdout. Refuses output carrying no `ws://`
 *  line by naming the log, rather than passing an empty string to `--connect` and letting the verify
 *  fail as though the adapter were at fault. */
export function readEndpoint(output: string, logPath: string): string {
    const line = output.split("\n").find((l) => l.trim().startsWith("ws://"));
    if (!line) throw new Error(`browser server printed no ws:// endpoint (see ${logPath})`);
    return line.trim();
}

/** Launch a real-GPU browser server from `app`'s OWN playwright and return its ws endpoint. The
 *  script is written into the app so `import "playwright"` resolves to the pinned 1.62.1 the previous
 *  install carries: the connect protocol answers a version mismatch with `428 Precondition Required`,
 *  so the repo's own playwright is the wrong module here even when the versions happen to agree. */
export async function startBrowserServer(
    app: string,
    evidence: string,
): Promise<{ endpoint: string; stop(): void }> {
    const script = join(app, "server.mjs");
    writeFileSync(
        script,
        'import { chromium } from "playwright";\n' +
            "const s = await chromium.launchServer(JSON.parse(process.argv[2]));\n" +
            "console.log(s.wsEndpoint());\n",
    );
    const logPath = join(evidence, "previous-browser-server.log");
    const proc = Bun.spawn(["bun", script, JSON.stringify(CONNECT_LAUNCH)], {
        cwd: app,
        stdout: "pipe",
        stderr: "pipe",
    });
    const stop = () => proc.kill();
    const decoder = new TextDecoder();
    const reader = proc.stdout.getReader();
    let out = "";
    try {
        const deadline = Bun.sleep(30_000).then(() => "timeout" as const);
        while (!out.includes("ws://")) {
            const next = await Promise.race([reader.read(), deadline]);
            if (next === "timeout" || next.done) break;
            out += decoder.decode(next.value, { stream: true });
        }
    } finally {
        reader.releaseLock();
    }
    const stderr = await Promise.race([
        new Response(proc.stderr).text(),
        Bun.sleep(2000).then(() => ""),
    ]);
    writeFileSync(logPath, `${out}\n--- stderr ---\n${stderr}`);
    try {
        return { endpoint: readEndpoint(out, logPath), stop };
    } catch (e) {
        stop();
        throw new Error(`${(e as Error).message}\n${stderr.slice(-2000)}`);
    }
}
