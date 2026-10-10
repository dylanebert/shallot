import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "../browser.fixture";

const SUBJECT = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SUBJECT, "../..");

async function unusedPort(): Promise<number> {
    const server = createServer();
    return new Promise<number>((resolvePort, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            if (!address || typeof address === "string") {
                server.close();
                reject(new Error("could not allocate a preview port"));
                return;
            }
            server.close((error) => (error ? reject(error) : resolvePort(address.port)));
        });
    });
}

async function waitForPreview(url: string, child: ChildProcess, log: () => string): Promise<void> {
    const deadline = Date.now() + 20_000;
    while (Date.now() < deadline) {
        if (child.exitCode !== null)
            throw new Error(`packed project's Vite preview exited:\n${log()}`);
        try {
            const response = await fetch(url);
            if (response.ok) return;
        } catch {
            // Vite has not opened its socket yet.
        }
        await new Promise((done) => setTimeout(done, 100));
    }
    throw new Error(`packed project's Vite preview did not become ready:\n${log()}`);
}

test("a Vite project built from the packed Shallot captures through its public rendering export", async ({
    page,
}) => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-capture-"));
    const project = join(scratch, "project");
    const packageManifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const tarballName = `dylanebert-shallot-${packageManifest.version}.tgz`;
    mkdirSync(project);
    let preview: ChildProcess | undefined;
    let previewLog = "";

    try {
        execFileSync(
            "bun",
            ["pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            { cwd: ROOT, stdio: "pipe" },
        );
        const tarball = join(scratch, tarballName);
        if (!existsSync(tarball)) throw new Error(`packed tarball is missing: ${tarball}`);

        writeFileSync(
            join(project, "package.json"),
            JSON.stringify({
                name: "packed-capture-project",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:../${tarballName}`,
                    typegpu: "~0.12.5",
                    vite: "^8.3.0",
                },
            }),
        );
        writeFileSync(
            join(project, "vite.config.ts"),
            readFileSync(join(SUBJECT, "vite.config.ts"), "utf8"),
        );
        writeFileSync(
            join(project, "index.html"),
            `<!doctype html>
<html><head><meta charset="UTF-8"><title>Packed Shallot capture</title></head>
<body><canvas id="frame" width="1280" height="720"></canvas><output id="result"></output>
<script type="module" src="/src/main.ts"></script></body></html>
`,
        );
        mkdirSync(join(project, "src"));
        writeFileSync(
            join(project, "src/main.ts"),
            `import { captureFrame } from "@dylanebert/shallot/rendering";
const canvas = document.querySelector<HTMLCanvasElement>("#frame")!;
const context = canvas.getContext("2d")!;
context.fillStyle = "rgb(17, 83, 199)";
context.fillRect(0, 0, canvas.width, canvas.height);
const image = await captureFrame(canvas);
document.querySelector("#result")!.textContent = JSON.stringify({
    width: image.width,
    height: image.height,
    identity: image.identity,
    hasColor: image.rgba.some((value, index) => index % 4 !== 3 && value !== 0),
});
`,
        );

        execFileSync("bun", ["install", "--no-progress"], { cwd: project, stdio: "pipe" });
        execFileSync("bun", ["x", "vite", "build"], { cwd: project, stdio: "pipe" });

        const port = await unusedPort();
        const url = `http://127.0.0.1:${port}/`;
        preview = spawn(
            "bun",
            ["x", "vite", "preview", "--host", "127.0.0.1", "--port", String(port), "--strictPort"],
            { cwd: project, stdio: ["ignore", "pipe", "pipe"] },
        );
        preview.stdout?.on("data", (chunk: Buffer) => (previewLog += chunk.toString()));
        preview.stderr?.on("data", (chunk: Buffer) => (previewLog += chunk.toString()));
        await waitForPreview(url, preview, () => previewLog);

        await page.goto(url);
        await expect(page.locator("#result")).not.toHaveText("");
        const result = JSON.parse((await page.locator("#result").textContent()) ?? "null") as {
            width: number;
            height: number;
            hasColor: boolean;
            identity: { width: number; height: number; surface: string; encoding: string };
        };
        expect(result.width).toBe(1280);
        expect(result.height).toBe(720);
        expect(result.identity).toMatchObject({
            width: 1280,
            height: 720,
            surface: "final-canvas",
            encoding: "rgba8-tight",
        });
        expect(result.hasColor).toBe(true);
    } finally {
        if (preview && preview.exitCode === null) {
            preview.kill("SIGTERM");
            await new Promise<void>((done) => preview?.once("exit", () => done()));
        }
        rmSync(scratch, { recursive: true, force: true });
    }
});
