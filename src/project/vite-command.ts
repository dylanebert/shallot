import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { dirname, resolve } from "node:path";

export type ViteCommand = "dev" | "build" | "preview";

/** Resolve the Vite CLI from this project without allowing a package-manager fetch. */
export function resolveViteCli(projectDir: string): string {
    const base = resolve(projectDir);
    let current = base;
    let packagePath: string | undefined;
    while (true) {
        const candidate = resolve(current, "node_modules", "vite", "package.json");
        if (existsSync(candidate)) {
            packagePath = candidate;
            break;
        }
        const parent = dirname(current);
        if (parent === current) break;
        current = parent;
    }
    if (packagePath === undefined)
        throw new Error(
            `Cannot resolve Vite from ${base}. Add vite as a devDependency with \`bun add -d vite\`.`,
        );
    const pkg = JSON.parse(readFileSync(packagePath, "utf8")) as {
        bin?: string | Record<string, string>;
    };
    const bin = typeof pkg.bin === "string" ? pkg.bin : pkg.bin?.vite;
    if (!bin) throw new Error(`Vite package at ${packagePath} does not declare its CLI binary.`);
    return resolve(dirname(packagePath), bin);
}

/** Run the Vite CLI resolved from the project and preserve its arguments and exit status. */
export function runVite(
    projectDir: string,
    command: ViteCommand,
    args: string[] = [],
    spawnSync: typeof Bun.spawnSync = Bun.spawnSync,
): number {
    let viteCli: string;
    try {
        viteCli = resolveViteCli(projectDir);
    } catch (error) {
        console.error(error instanceof Error ? error.message : String(error));
        return 1;
    }
    const result = spawnSync([process.execPath, viteCli, ...viteCliArgs(command, args)], {
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

interface ReportedViteUrl {
    kind: "local" | "network";
    url: string;
}

function reportedViteUrl(line: string): ReportedViteUrl | null {
    const clean = line.replace(/\u001b\[[0-9;]*m/g, "");
    const match = clean.match(/\b(Local|Network):\s*(https?:\/\/\S+)/);
    if (!match) return null;
    return { kind: match[1].toLowerCase() as ReportedViteUrl["kind"], url: match[2] };
}

export function viteUrlFromLines(lines: readonly string[]): string | null {
    let network: string | null = null;
    for (const line of lines) {
        const reported = reportedViteUrl(line);
        if (reported?.kind === "local") return reported.url;
        if (reported?.kind === "network" && network === null) network = reported.url;
    }
    return network;
}

export interface ViteDevProcess {
    url: string;
    close(): Promise<void>;
}

/** Start the project's own Vite dev server and report its actual local URL. */
export async function startViteDev(
    projectDir: string,
    args: string[] = [],
    viteCli = resolveViteCli(projectDir),
): Promise<ViteDevProcess> {
    const child = spawn(process.execPath, [viteCli, ...args], {
        cwd: projectDir,
        stdio: ["inherit", "pipe", "pipe"],
    });
    const output: string[] = [];
    let networkFallback: ReturnType<typeof setTimeout> | undefined;
    const clearNetworkFallback = () => {
        if (networkFallback !== undefined) clearTimeout(networkFallback);
        networkFallback = undefined;
    };
    const lines = (stream: NodeJS.ReadableStream, destination: NodeJS.WriteStream) => {
        const reader = createInterface({ input: stream });
        reader.on("line", (line) => {
            if (!settled) {
                output.push(line);
                if (output.length > 30) output.shift();
            }
            destination.write(`${line}\n`);
            const report = reportedViteUrl(line);
            if (report?.kind === "local") {
                ready(viteUrlFromLines(output) ?? report.url);
            } else if (report?.kind === "network" && networkFallback === undefined) {
                networkFallback = setTimeout(() => {
                    const url = viteUrlFromLines(output);
                    if (url) ready(url);
                }, 50);
            }
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
        clearNetworkFallback();
        resolveReady(url);
    };
    lines(child.stdout, process.stdout);
    lines(child.stderr, process.stderr);
    child.once("error", (error) => {
        if (settled) return;
        settled = true;
        clearNetworkFallback();
        rejectReady(error);
    });
    child.once("exit", (code, signal) => {
        if (settled) return;
        settled = true;
        clearNetworkFallback();
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
