import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

export type ViteCommand = "dev" | "build" | "preview";

/** Run the Vite CLI resolved from the project and preserve its arguments and exit status. */
export function runVite(projectDir: string, command: ViteCommand, args: string[] = []): number {
    const result = Bun.spawnSync([process.execPath, "x", "vite", ...viteCliArgs(command, args)], {
        cwd: projectDir,
        stdin: "inherit",
        stdout: "inherit",
        stderr: "inherit",
    });
    return result.exitCode;
}

export function viteCliArgs(command: ViteCommand, args: string[] = []): string[] {
    return [...(command === "dev" ? [] : [command]), ...args];
}

export interface ViteDevProcess {
    url: string;
    close(): Promise<void>;
}

/** Start the project's own Vite dev server and report its actual local URL. */
export async function startViteDev(
    projectDir: string,
    args: string[] = [],
): Promise<ViteDevProcess> {
    const child = spawn(process.execPath, ["x", "vite", ...args], {
        cwd: projectDir,
        stdio: ["inherit", "pipe", "pipe"],
    });
    const output: string[] = [];
    const lines = (stream: NodeJS.ReadableStream, destination: NodeJS.WriteStream) => {
        const reader = createInterface({ input: stream });
        reader.on("line", (line) => {
            if (!settled) {
                output.push(line);
                if (output.length > 30) output.shift();
            }
            destination.write(`${line}\n`);
            const match = line.match(/https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?\/?/);
            if (match) ready(match[0]);
        });
    };

    let resolveReady!: (url: string) => void;
    let rejectReady!: (error: Error) => void;
    const readyUrl = new Promise<string>((resolve, reject) => {
        resolveReady = resolve;
        rejectReady = reject;
    });
    let settled = false;
    const ready = (url: string) => {
        if (settled) return;
        settled = true;
        resolveReady(url);
    };
    lines(child.stdout, process.stdout);
    lines(child.stderr, process.stderr);
    child.once("error", (error) => {
        if (settled) return;
        settled = true;
        rejectReady(error);
    });
    child.once("exit", (code, signal) => {
        if (settled) return;
        settled = true;
        rejectReady(
            new Error(
                `Vite exited before reporting its dev URL (${signal ?? `code ${code}`})${output.length ? `:\n${output.join("\n")}` : ""}`,
            ),
        );
    });

    const url = await readyUrl;
    return {
        url,
        close: async () => {
            if (child.exitCode !== null || child.signalCode !== null) return;
            await new Promise<void>((resolve) => {
                child.once("exit", () => resolve());
                child.kill("SIGTERM");
            });
        },
    };
}
