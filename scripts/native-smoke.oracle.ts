import { expect, setDefaultTimeout, test } from "bun:test";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import {
    chmodSync,
    copyFileSync,
    existsSync,
    mkdtempSync,
    readFileSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { cargoBuildEnvironment, resolveCargoInvocation } from "../src/native";
import {
    assertCanvasImage,
    assertNoSmokeProcesses,
    assertStandaloneOrigin,
    assertStartup,
    assertSystemOrigin,
    macApp,
    macOwnedPids,
    macSmokePids,
    requireLinuxPrerequisites,
    standaloneIndex,
    startObserver,
    startVirtualDisplay,
    stopMainShell,
    streamLines,
    terminateProcessGroup,
    windowsOutput,
    windowsSmokePids,
    withoutProject,
} from "./native-smoke";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

test("the final inventory refuses leftover smoke processes", async () => {
    expect(() => assertNoSmokeProcesses([])).not.toThrow();
    expect(() => assertNoSmokeProcesses([80337])).toThrow("smoke processes still running: 80337");
    if (process.platform !== "darwin") return;
    const owner = `SHALLOT_SMOKE_OWNER=oracle-${crypto.randomUUID()}`;
    const child = spawn(process.execPath, ["-e", 'console.log("ready");setInterval(()=>{},1000)'], {
        env: { ...process.env, SHALLOT_SMOKE_OWNER: owner.split("=")[1] },
        stdio: ["ignore", "pipe", "inherit"],
    });
    try {
        await once(child.stdout!, "data");
        const pids = macOwnedPids(owner);
        expect(pids).toContain(child.pid!);
        expect(() => assertNoSmokeProcesses(pids)).toThrow("smoke processes still running");
    } finally {
        child.kill();
        await once(child, "close");
    }
    expect(macOwnedPids(owner)).toEqual([]);
});

test("Mac bundle paths distinguish backend, profile and dev assets", () => {
    expect(macApp("/scratch/game", false, false)).toBe(
        resolve("/scratch/game/build/mac/debug-system/game.app"),
    );
    expect(macApp("/scratch/game", true, true)).toBe(
        resolve("/scratch/game/build/mac/release-portable/game.app"),
    );
    expect(macApp("/scratch/game", true, false, true)).toBe(
        resolve("/scratch/game/build/mac/debug-portable-dev/game.app"),
    );
});

test("a missing-asset diagnosis checks the launched bundle on each host", () => {
    const app = macApp("/scratch/game", true, false);
    expect(
        standaloneIndex("/scratch/game", "debug-preview", {
            executable: `${app}/Contents/MacOS/game`,
        }),
    ).toBe(resolve(app, "Contents/Resources/dist/index.html"));
    expect(standaloneIndex("/scratch/game", "debug-preview")).toBe(
        resolve("/scratch/game/build/linux/debug-portable/dist/index.html"),
    );
    expect(standaloneIndex("/scratch/game", "release-preview")).toBeNull();
});

test("system previews require the shell custom protocol, not Vite or a file", () => {
    expect(() => assertSystemOrigin("shallot://localhost/")).not.toThrow();
    for (const href of ["http://localhost:5173/", "file:///tmp/index.html", "shallot://other/"])
        expect(() => assertSystemOrigin(href)).toThrow(/not shallot/);
});

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

test("default native startup needs observed CPU startup and splash refusal, not a software canvas", () => {
    const report = {
        requireHardware: true,
        cpuReady: true,
        hooks: [],
        refusal: "Hardware WebGPU is required: fallback adapter SwiftShader",
    };
    expect(() => assertStartup(report)).not.toThrow();
    expect(() => assertStartup({ ...report, refusal: undefined })).toThrow(
        /refusal was not observed/,
    );
    expect(() => assertStartup({ ...report, refusal: "generic failure" })).toThrow(
        /refusal was not observed/,
    );
    expect(() => assertStartup({ ...report, hooks: ["prepare"] })).toThrow(/before plugin hooks/);
    expect(() => assertStartup({ ...report, cpuReady: false })).toThrow(
        /CPU-only startup was not observed/,
    );
    expect(() => assertStartup({ ...report, error: "page failed" })).toThrow(/page failed/);
    expect(() => assertStartup({ ...report, requireHardware: undefined })).toThrow(
        /policy was not published/,
    );
});

test("deliberate native software startup still requires plugin hooks and a green canvas", () => {
    const report = {
        requireHardware: false,
        cpuReady: true,
        hooks: ["prepare", "initialize", "warm"],
    };
    expect(() => assertStartup(report)).toThrow(/WebGPU canvas image failed/);
    expect(() => assertStartup({ ...report, pixel: [0, 255, 0, 255] })).not.toThrow();
    expect(() => assertStartup({ ...report, hooks: [] })).toThrow(
        /software startup was not observed/,
    );
});

test("hardware smoke requires an identified real adapter, hooks and canvas under the default policy", () => {
    const report = {
        requireHardware: true,
        cpuReady: true,
        hooks: ["prepare", "initialize", "warm"],
        pixel: [0, 255, 0, 255],
        adapter: { vendor: "nvidia", architecture: "lovelace", isFallbackAdapter: false },
    };
    expect(() => assertStartup(report, true)).not.toThrow();
    const refusedAdapters: (Record<string, string | boolean> | undefined)[] = [
        undefined,
        {},
        { description: "SwiftShader" },
        { vendor: "nvidia", isFallbackAdapter: true },
    ];
    for (const adapter of refusedAdapters)
        expect(() => assertStartup({ ...report, adapter }, true)).toThrow(
            /Hardware adapter required/,
        );
    expect(() => assertStartup({ ...report, hooks: [] }, true)).toThrow(/Hardware startup/);
    expect(() => assertStartup({ ...report, requireHardware: false }, true)).toThrow(
        /default native GPU policy/,
    );
    expect(() => assertStartup({ ...report, pixel: [] }, true)).toThrow(/canvas image failed/);
    expect(
        requireLinuxPrerequisites(
            {
                commandExists: () => true,
                hasX11Headers: () => true,
                softwareVulkanIcd: () => null,
            },
            true,
        ),
    ).toBe("");
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

if (process.platform === "linux") {
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
            timeoutMs: 15_000,
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
                startVirtualDisplay(
                    scratch,
                    { ...env, MODE: "silent" },
                    { ...probe, timeoutMs: 300 },
                ),
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
}

test("native Windows overrides the cross linker even when LLVM is not installed", () => {
    const target = "x86_64-pc-windows-msvc";
    const key = "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER";
    const config = Bun.TOML.parse(
        readFileSync(resolve(import.meta.dir, "../crates/native/.cargo/config.toml"), "utf8"),
    ) as { target: Record<string, { linker: string }> };
    for (const portable of [false, true]) {
        const invocation = resolveCargoInvocation(target, false, portable, false, "win32");
        const env = cargoBuildEnvironment(target, invocation, { [key]: "lld-link" });
        expect(env[key] ?? config.target[target].linker).toBe("link.exe");
    }
    for (const wsl of [false, true]) {
        const invocation = resolveCargoInvocation(target, false, false, wsl, "linux");
        const env = cargoBuildEnvironment(target, invocation, {});
        expect(env[key] ?? config.target[target].linker).toBe("lld-link");
    }
    const linux = resolveCargoInvocation("x86_64-unknown-linux-gnu", false, true, false, "linux");
    expect(
        cargoBuildEnvironment("x86_64-unknown-linux-gnu", linux, {})
            .CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER,
    ).toBeUndefined();
});

test("Windows builds use native MSVC rather than cargo-xwin", () => {
    expect(
        resolveCargoInvocation("x86_64-pc-windows-msvc", false, false, false, "win32").kind,
    ).toBe("native");
    expect(resolveCargoInvocation("x86_64-pc-windows-msvc", false, true, false, "win32").kind).toBe(
        "native",
    );
    expect(
        resolveCargoInvocation("x86_64-pc-windows-msvc", false, false, false, "linux").kind,
    ).toBe("xwin");
});

test("Windows output paths distinguish backend, profile and dev", () => {
    expect(windowsOutput("/scratch/game", true, false, true)).toBe(
        resolve("/scratch/game/build/windows/debug-portable-dev"),
    );
    expect(windowsOutput("/scratch/game", false, true)).toBe(
        resolve("/scratch/game/build/windows/release-system"),
    );
});

function running(pid: number): boolean {
    if (process.platform === "win32") {
        try {
            process.kill(pid, 0);
            return true;
        } catch {
            return false;
        }
    }
    if (process.platform === "darwin") return spawnSync("ps", ["-p", String(pid)]).status === 0;
    try {
        const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
        const state = stat.slice(stat.lastIndexOf(")") + 2, stat.lastIndexOf(")") + 3);
        return state !== "Z" && state !== "X";
    } catch {
        return false;
    }
}

test("native smoke cleanup terminates the whole detached process group", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-smoke-process-tree-"));
    const ticks = join(scratch, "ticks");
    const childCode = `const {spawn}=require("node:child_process");const grandchild=spawn(process.execPath,["-e",${JSON.stringify(`const fs=require("node:fs");let ready=false;setInterval(()=>{fs.appendFileSync(${JSON.stringify(ticks)},"tick\\n");if(!ready){ready=true;console.log("ready");}},25);`)}],{stdio:["ignore","pipe","inherit"]});grandchild.stdout.once("data",()=>console.log(process.pid+" "+grandchild.pid));setInterval(()=>{},1000);`;
    const parent = spawn(process.execPath, ["-e", childCode], {
        detached: true,
        stdio: ["ignore", "pipe", "pipe"],
    });
    const ready = once(parent.stdout!, "data");
    const exited = once(parent, "close");
    try {
        if (!parent.pid) throw new Error("test parent process did not start");
        const event = await Promise.race([
            ready.then(([data]) => String(data)),
            exited.then(() => {
                throw new Error("fixture exited before readiness");
            }),
        ]);
        const [parentPid, grandchildPid] = event.trim().split(" ").map(Number);
        expect(existsSync(ticks)).toBe(true);
        await terminateProcessGroup(parent.pid, 200);
        await exited;
        const before = readFileSync(ticks, "utf8");
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

test("shell cleanup finds the copied executable even outside the launcher group on Mac", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-shell-"));
    const executable = join(
        scratch,
        process.platform === "win32" ? "smoke-sleeper.exe" : "smoke-sleeper",
    );
    const source = join(scratch, "sleeper.c");
    writeFileSync(
        source,
        '#include <unistd.h>\nint main(void) { write(1, "ready\\n", 6); for (;;) pause(); }\n',
    );
    if (process.platform === "win32") copyFileSync(process.execPath, executable);
    else {
        const compiled = spawnSync("cc", [source, "-o", executable], { encoding: "utf8" });
        if (compiled.status !== 0)
            throw new Error(`fixture compilation failed: ${compiled.stderr}`);
    }
    const child = spawn(
        executable,
        process.platform === "win32" ? ["-e", 'console.log("ready");setInterval(()=>{},1000)'] : [],
        { detached: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    const ready = once(child.stdout!, "data");
    const exited = once(child, "close");
    try {
        if (!child.pid) throw new Error("fixture did not start");
        const result = await Promise.race([ready.then(() => "ready"), exited.then(() => "exited")]);
        expect(result).toBe("ready");
        expect(running(child.pid)).toBe(true);
        if (process.platform === "win32") {
            expect(windowsSmokePids(scratch)).toContain(child.pid);
            expect(() => assertNoSmokeProcesses(windowsSmokePids(scratch))).toThrow(
                "smoke processes still running",
            );
        }
        if (process.platform === "darwin") {
            expect(macSmokePids(scratch)).toContain(child.pid);
            expect(() => assertNoSmokeProcesses(macSmokePids(scratch))).toThrow(
                "smoke processes still running",
            );
        }
        await stopMainShell(process.platform === "darwin" ? process.pid : child.pid, executable);
        await exited;
        expect(running(child.pid)).toBe(false);
        if (process.platform === "darwin") assertNoSmokeProcesses(macSmokePids(scratch));
    } finally {
        if (child.pid) await terminateProcessGroup(child.pid);
        removeTestDirectory(scratch);
    }
});

test("copied launches cannot read source assets, and restoration survives a rejected dependency", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-unavailable-"));
    const project = join(scratch, "project");
    writeFileSync(project, "source assets");
    try {
        await expect(
            withoutProject(project, async () => {
                expect(existsSync(project)).toBe(false);
                readFileSync(project);
            }),
        ).rejects.toThrow(/ENOENT/);
        expect(readFileSync(project, "utf8")).toBe("source assets");
    } finally {
        removeTestDirectory(scratch);
    }
});

test("each preview of the same artifact needs a fresh page-run marker", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-marker-"));
    const observer = await startObserver(scratch);
    const send = (run: string) =>
        fetch(`${observer.origin}/marker`, {
            method: "POST",
            body: JSON.stringify({
                nonce: "artifact",
                run,
                mode: "release-preview",
                href: "shallot://localhost/",
            }),
        });
    try {
        const first = observer.wait("artifact");
        expect((await send("preview-run")).status).toBe(200);
        expect((await first).run).toBe("preview-run");
        const copied = observer.wait("artifact");
        expect((await send("preview-run")).status).toBe(409);
        expect((await send("copied-run")).status).toBe(200);
        expect((await copied).run).toBe("copied-run");
    } finally {
        await observer.close();
        removeTestDirectory(scratch);
    }
});

test("a page that never reports names the stages it reached for this launch only", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-stage-"));
    const observer = await startObserver(scratch);
    const post = (stage: string) =>
        fetch(`${observer.origin}/stage`, {
            method: "POST",
            body: JSON.stringify({ nonce: "artifact", stage }),
        });
    try {
        void observer.wait("artifact");
        expect((await post("html")).status).toBe(204);
        expect((await post("module")).status).toBe(204);
        expect(observer.stages("artifact")).toEqual(["html", "module"]);
        void observer.wait("artifact");
        expect(observer.stages("artifact")).toEqual([]);
    } finally {
        await observer.close();
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
