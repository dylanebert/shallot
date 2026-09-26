import { expect } from "bun:test";
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { main } from "./index";
import { startWebPreview } from "./preview";

class ExitStatus extends Error {
    constructor(readonly code: number) {
        super(`exit ${code}`);
    }
}

function snapshot(directory: string): Record<string, string> {
    return Object.fromEntries(
        readdirSync(directory)
            .sort()
            .map((name) => [name, readFileSync(join(directory, name)).toString("base64")]),
    );
}

check(
    "preview refuses a missing web build",
    {
        claim: "shallot preview refuses when its web build is missing and tells the user to run shallot build",
    },
    async () => {
        const project = mkdtempSync(join(tmpdir(), "shallot-preview-missing-"));
        const errors: string[] = [];
        const originalError = console.error;
        let code = 0;
        console.error = (...values: unknown[]) => errors.push(values.join(" "));
        try {
            try {
                await main(["preview", project], (status) => {
                    throw new ExitStatus(status);
                });
            } catch (error) {
                if (!(error instanceof ExitStatus)) throw error;
                code = error.code;
            }
            expect(code).toBe(1);
            expect(errors.join("\n")).toContain("no web build found at dist/index.html");
            expect(errors.join("\n")).toContain("run `shallot build` first");
            expect(existsSync(join(project, "dist"))).toBe(false);

            errors.length = 0;
            code = 0;
            try {
                await main(["preview", project, "--target", "linux"], (status) => {
                    throw new ExitStatus(status);
                });
            } catch (error) {
                if (!(error instanceof ExitStatus)) throw error;
                code = error.code;
            }
            expect(code).toBe(1);
            expect(errors.join("\n")).toContain("no linux build found");
            expect(errors.join("\n")).toContain("run `shallot build --target linux` first");
        } finally {
            console.error = originalError;
            rmSync(project, { recursive: true, force: true });
        }
    },
);

check(
    "preview serves the existing web build without writing to it",
    {
        claim: "shallot preview serves the existing web build with cross-origin isolation and leaves its files unchanged",
    },
    async () => {
        const project = mkdtempSync(join(tmpdir(), "shallot-preview-web-"));
        const dist = join(project, "dist");
        mkdirSync(dist);
        writeFileSync(join(dist, "index.html"), "<!doctype html><title>built page</title>\n");
        writeFileSync(join(dist, "game.js"), "export const built = true;\n");
        const before = snapshot(dist);
        let server: Awaited<ReturnType<typeof startWebPreview>> | undefined;
        try {
            server = await startWebPreview(project, { port: 0, open: false, host: "127.0.0.1" });
            const address = server.httpServer.address();
            if (address === null || typeof address === "string")
                throw new Error("preview server has no TCP address");
            const response = await fetch(`http://127.0.0.1:${address.port}/`);
            expect(response.status).toBe(200);
            expect(await response.text()).toBe("<!doctype html><title>built page</title>\n");
            expect(response.headers.get("Cross-Origin-Opener-Policy")).toBe("same-origin");
            expect(response.headers.get("Cross-Origin-Embedder-Policy")).toBe("require-corp");
            expect(snapshot(dist)).toEqual(before);
        } finally {
            await server?.close();
            rmSync(project, { recursive: true, force: true });
        }
    },
);
