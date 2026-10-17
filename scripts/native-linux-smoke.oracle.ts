import { expect, setDefaultTimeout, test } from "bun:test";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    assertCanvasImage,
    assertStandaloneOrigin,
    requireLinuxPrerequisites,
    startVirtualDisplay,
    streamLines,
    terminateProcessGroup,
} from "./native-linux-smoke";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

test("Linux smoke refuses a missing system prerequisite with the setup command", () => {
    expect(() =>
        requireLinuxPrerequisites({
            commandExists: (name) => name !== "Xvfb",
            hasX11Headers: () => true,
            softwareVulkanIcd: () => "/mesa/lvp_icd.json",
        }),
    ).toThrow(/Xvfb.*system packages: `bash scripts\/native-linux-setup\.sh`/);
});

test("Linux smoke requires the compositor that exposes the X11 visual mismatch", () => {
    expect(() =>
        requireLinuxPrerequisites({
            commandExists: (name) => name !== "xcompmgr",
            hasX11Headers: () => true,
            softwareVulkanIcd: () => "/mesa/lvp_icd.json",
        }),
    ).toThrow(/xcompmgr/);
});

test("an executed page is insufficient without the WebGPU canvas image", () => {
    expect(() => assertCanvasImage({})).toThrow(/WebGPU canvas image failed/);
    expect(() => assertCanvasImage({ pixel: [0, 0, 0, 0] })).toThrow(/WebGPU canvas image failed/);
    expect(() => assertCanvasImage({ pixel: [0, 255, 0, 255], error: "device lost" })).toThrow(
        /device lost/,
    );
    expect(() => assertCanvasImage({ pixel: [0, 255, 0, 255] })).not.toThrow();
});

test("Linux smoke names the Bun and Rust toolchain installers", () => {
    const refuse = () =>
        requireLinuxPrerequisites({
            commandExists: (name) => !["bun", "bunx", "cargo", "rustc"].includes(name),
            hasX11Headers: () => true,
            softwareVulkanIcd: () => "/mesa/lvp_icd.json",
        });
    expect(refuse).toThrow(/Bun: `curl -fsSL https:\/\/bun\.sh\/install \| bash`/);
    expect(refuse).toThrow(
        /Rust: `curl --proto '=https' --tlsv1\.2 -sSf https:\/\/sh\.rustup\.rs \| sh`/,
    );
});

test("standalone pages require the exact observed shell asset-server origin", () => {
    expect(() => assertStandaloneOrigin("http://127.0.0.1:4173/", null, null)).toThrow(
        /did not report its asset-server origin/,
    );
    expect(() =>
        assertStandaloneOrigin("http://127.0.0.1:4174/", "http://127.0.0.1:4173", null),
    ).toThrow(/not the shell asset server http:\/\/127\.0\.0\.1:4173/);
    expect(() =>
        assertStandaloneOrigin("http://127.0.0.1:4173/", "http://127.0.0.1:4173", null),
    ).not.toThrow();
});

test("virtual display skips occupied locks and sockets and waits for server notification", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-display-contract-"));
    const executable = join(scratch, "Xvfb");
    const record = join(scratch, "record");
    writeFileSync(
        executable,
        `#!/usr/bin/python3
import os, signal, sys, time
assert sys.argv[1:] == [':101', '-screen', '0', '1280x720x24', '-nolisten', 'tcp', '-ac'], sys.argv
assert signal.getsignal(signal.SIGUSR1) == signal.SIG_IGN
open(os.environ['RECORD'], 'w').write(str(os.getpid()))
print('101', flush=True)
if os.environ['MODE'] == 'refuse':
    print('fixture refusal', file=sys.stderr, flush=True)
    sys.exit(7)
if os.environ['MODE'] == 'ready':
    os.kill(os.getppid(), signal.SIGUSR1)
while True: time.sleep(1)
`,
    );
    chmodSync(executable, 0o755);
    const inspected: string[] = [];
    const probe = {
        pathExists(path: string) {
            inspected.push(path);
            return path === "/tmp/.X99-lock" || path === "/tmp/.X11-unix/X100";
        },
        timeoutMs: 300,
    };
    const env = { ...process.env, PATH: `${scratch}:${process.env.PATH}`, RECORD: record };
    try {
        const display = await startVirtualDisplay(scratch, { ...env, MODE: "ready" }, probe);
        expect(display.display).toBe(":101");
        const pid = Number(readFileSync(record, "utf8"));
        expect(running(pid)).toBe(true);
        await display.stop();
        expect(running(pid)).toBe(false);
        expect(inspected).toEqual([
            "/tmp/.X99-lock",
            "/tmp/.X100-lock",
            "/tmp/.X11-unix/X100",
            "/tmp/.X101-lock",
            "/tmp/.X11-unix/X101",
        ]);
        await expect(
            startVirtualDisplay(scratch, { ...env, MODE: "refuse" }, probe),
        ).rejects.toThrow(/Xvfb exited \(7\)/);
        expect(readFileSync(join(scratch, "virtual-display.log"), "utf8")).toContain(
            "fixture refusal",
        );
        await expect(
            startVirtualDisplay(scratch, { ...env, MODE: "silent" }, probe),
        ).rejects.toThrow(/did not signal readiness/);
        expect(running(Number(readFileSync(record, "utf8")))).toBe(false);
        await expect(
            startVirtualDisplay(scratch, env, {
                pathExists: () => true,
                timeoutMs: 300,
            }),
        ).rejects.toThrow(/no unused display number/);
    } finally {
        removeTestDirectory(scratch);
    }
});

function running(pid: number): boolean {
    try {
        const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
        const state = stat.slice(stat.lastIndexOf(")") + 2, stat.lastIndexOf(")") + 3);
        return state !== "Z" && state !== "X";
    } catch {
        return false;
    }
}

test("Linux smoke cleanup terminates the whole detached process group", async () => {
    if (process.platform !== "linux") throw new Error("this process-tree assertion is Linux-only");
    const scratch = mkdtempSync(join(tmpdir(), "shallot-smoke-process-tree-"));
    const pidFile = join(scratch, "pids");
    const ticks = join(scratch, "ticks");
    const childCode = `const fs=require("node:fs");const {spawn}=require("node:child_process");const grandchild=spawn(process.execPath,["-e",${JSON.stringify(`const fs=require("node:fs");setInterval(()=>fs.appendFileSync(${JSON.stringify(ticks)},"tick\\n"),25);`)}],{stdio:"ignore"});fs.writeFileSync(${JSON.stringify(pidFile)},process.pid+" "+grandchild.pid);setInterval(()=>{},1000);`;
    const parent = spawn(process.execPath, ["-e", childCode], {
        detached: true,
        stdio: "ignore",
    });
    try {
        if (!parent.pid) throw new Error("test parent process did not start");
        const deadline = Date.now() + 5000;
        while (!existsSync(pidFile) && Date.now() < deadline) await Bun.sleep(20);
        expect(existsSync(pidFile)).toBe(true);
        const [parentPid, grandchildPid] = readFileSync(pidFile, "utf8").split(" ").map(Number);
        const tickDeadline = Date.now() + 2000;
        while (
            (!existsSync(ticks) || readFileSync(ticks, "utf8").split("\n").length < 3) &&
            Date.now() < tickDeadline
        )
            await Bun.sleep(20);
        expect(existsSync(ticks)).toBe(true);
        const before = readFileSync(ticks, "utf8");
        await terminateProcessGroup(parent.pid, 200);
        await Promise.race([once(parent, "close"), Bun.sleep(1500)]);
        await Bun.sleep(100);
        const after = readFileSync(ticks, "utf8");
        expect(after).toBe(before);
        expect(running(parentPid)).toBe(false);
        expect(running(grandchildPid)).toBe(false);
    } finally {
        if (parent.pid) await terminateProcessGroup(parent.pid, 100);
        removeTestDirectory(scratch);
    }
});

test("the smoke log keeps a line whole when the shell writes it in pieces", () => {
    const written: string[] = [];
    const lines = streamLines((stream, line) => written.push(`[${stream}] ${line}`));
    lines.push("stderr", "[shallot] asset server on ");
    lines.push("stdout", "running game...\n");
    lines.push("stderr", "http://127.0.0.1:41301\nnext");
    lines.flush();
    expect(written).toEqual([
        "[stdout] running game...",
        "[stderr] [shallot] asset server on http://127.0.0.1:41301",
        "[stderr] next",
    ]);
});
