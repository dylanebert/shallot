import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { createServer } from "vite";
import { devConfig } from "./dev";

check(
    "dev serves a project-owned index page",
    {
        claim: "shallot dev serves the project's own index.html instead of synthesizing a game canvas page",
        size: "integration",
        subject: "src/cli/dev.ts",
        budget: 10_000,
    },
    async () => {
        const project = mkdtempSync(join(tmpdir(), "shallot-dev-owned-index-"));
        writeFileSync(
            join(project, "shallot.json"),
            JSON.stringify({
                kind: "recipe",
                description: "A page-owned project fixture.",
                problem: "I want my own host page.",
                plugins: {},
            }),
        );
        writeFileSync(
            join(project, "index.html"),
            "<!doctype html><html><body><main>PROJECT INDEX SENTINEL</main></body></html>\n",
        );
        const serveRoot = async (): Promise<string> => {
            const server = await createServer(
                devConfig(project, "owned-index", { port: 0, strictPort: true, open: false }),
            );
            try {
                await server.listen();
                const address = server.httpServer?.address();
                if (address === undefined || address === null || typeof address === "string")
                    throw new Error("dev server has no TCP address");
                const host = address.family === "IPv6" ? `[${address.address}]` : address.address;
                const response = await fetch(`http://${host}:${address.port}/`);
                expect(response.status).toBe(200);
                return response.text();
            } finally {
                await server.close();
            }
        };
        try {
            const ownedPage = await serveRoot();
            expect(ownedPage).toContain("PROJECT INDEX SENTINEL");
            expect(ownedPage).not.toContain('<canvas id="canvas"></canvas>');

            rmSync(join(project, "index.html"));
            const synthesizedPage = await serveRoot();
            expect(synthesizedPage).toContain('<canvas id="canvas"></canvas>');
        } finally {
            rmSync(project, { recursive: true, force: true });
        }
    },
);
