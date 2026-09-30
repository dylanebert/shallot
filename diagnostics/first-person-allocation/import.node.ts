import { setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

test("a non-page allocation row can import the allocation instrument without loading Vite before requesting a page build", () => {
    const probe = `
            import { plugin } from "bun";
            globalThis.viteLoads = [];
            plugin({
                name: "allocation-vite-load-observer",
                setup(build) {
                    build.onLoad({ filter: /node_modules[/\\\\]vite[/\\\\]/ }, async (args) => {
                        globalThis.viteLoads.push(args.path);
                        return { contents: await Bun.file(args.path).text(), loader: "js" };
                    });
                },
            });
            const { allocationFailure } = await import(${JSON.stringify(resolve(import.meta.dir, "allocation.ts"))});
            if (allocationFailure({ warm: 1, windows: [] }) === undefined)
                throw new Error("allocation export did not evaluate");
            if (globalThis.viteLoads.length !== 0)
                throw new Error("allocation import loaded Vite: " + globalThis.viteLoads.join(", "));
            await import("vite");
            if (globalThis.viteLoads.length === 0)
                throw new Error("Vite load observer did not detect the explicit import");
        `;
    const proc = Bun.spawnSync([process.execPath, "--eval", probe], {
        stdout: "pipe",
        stderr: "pipe",
        timeout: CEILING.startup,
    });
    if (proc.exitCode !== 0)
        throw new Error(`Bun allocation import failed: ${proc.stderr.toString()}`);
});
