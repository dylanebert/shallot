import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { projectPlugin } from "./vite";

check(
    "project plugin: dependency config is shared",
    {
        claim: "a project build bundles a second copy of the engine, typegpu or a manifest package plugin, splitting their module state",
    },
    () => {
        const root = mkdtempSync(join(tmpdir(), "shallot-project-plugin-"));
        try {
            writeFileSync(
                join(root, "shallot.json"),
                JSON.stringify({
                    plugins: {
                        Grid: "@dylanebert/shallot-grid/core",
                        Local: "./src/local.ts",
                    },
                }),
            );
            const plugin = projectPlugin(root);
            const configHook = plugin.config as unknown as (
                config: unknown,
                env: unknown,
            ) => unknown;
            const config = configHook({}, {}) as {
                resolve?: { dedupe?: string[] };
                optimizeDeps?: { exclude?: string[] };
            };
            const expected = ["@dylanebert/shallot", "typegpu", "@dylanebert/shallot-grid"];
            expect(config.resolve?.dedupe).toEqual(expected);
            expect(config.optimizeDeps?.exclude).toEqual(expected);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);
