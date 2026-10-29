import { expect, setDefaultTimeout, test } from "bun:test";
import { resolve } from "node:path";

import { TIMEOUT } from "@dylanebert/shallot/testing";
import { normalizePath } from "../../src/project/paths";

setDefaultTimeout(TIMEOUT.node);

/** A bundle input's absolute path with `/` separators, so suffix checks hold on Windows. */
const absolute = (path: string) => normalizePath(resolve(path));

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
        timeout: TIMEOUT.startup,
    });
    if (proc.exitCode !== 0)
        throw new Error(`Bun allocation import failed: ${proc.stderr.toString()}`);
});

test("a Node allocation import does not load the display-only oracle or Hyprland instrument", async () => {
    const built = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation.ts")],
        metafile: true,
        external: ["playwright", "webgpu", "chromium-bidi"],
        target: "bun",
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`allocation import graph failed: ${built.logs.map(String).join("\\n")}`);
    const inputs = Object.keys(built.metafile.inputs).map(absolute);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display.ts"))).toBe(false);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display-seat.ts"))).toBe(false);

    const node = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation-sampler.mjs")],
        metafile: true,
        target: "node",
        external: ["webgpu"],
    });
    if (!node.success || node.metafile === undefined)
        throw new Error(`Node sampler graph failed: ${node.logs.map(String).join("\\n")}`);
    const nodeInputs = Object.keys(node.metafile.inputs).map(absolute);
    expect(nodeInputs.some((path) => path.endsWith("/allocation-sampler.mjs"))).toBe(true);
    expect(nodeInputs.some((path) => path.endsWith("/display.ts"))).toBe(false);
    const nodeImports = Object.values(node.metafile.inputs).flatMap((input) => input.imports);
    expect(nodeImports.some((edge) => edge.external && edge.path === "webgpu")).toBe(true);
    expect(nodeImports.some((edge) => edge.external && /^(?:bun:|playwright)/.test(edge.path))).toBe(
        false,
    );
});
