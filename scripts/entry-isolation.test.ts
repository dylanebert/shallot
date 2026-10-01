import { expect, test } from "bun:test";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const BUILTINS = /^(?:node:|bun:|fs$|path$|os$|child_process$)/;

function checkBuild(result: Awaited<ReturnType<typeof Bun.build>>, label: string) {
    if (!result.success || result.metafile === undefined) {
        throw new Error(`${label} bundle failed:\n${result.logs.map(String).join("\n")}`);
    }
    return result.metafile;
}

function edgesOf(metafile: { inputs: Record<string, { imports: { path: string }[] }> }) {
    return Object.values(metafile.inputs).flatMap((input) =>
        input.imports.map((edge) => edge.path),
    );
}

test("browser exports load without Bun- or Node-only modules", async () => {
    const result = await Bun.build({
        entrypoints: [resolve(ROOT, "scripts/capture-page.ts")],
        target: "browser",
        format: "esm",
        outdir: resolve(ROOT, ".artifacts/entry-browser"),
        metafile: true,
    });
    const metafile = checkBuild(result, "browser capture");
    expect(edgesOf(metafile).filter((path) => BUILTINS.test(path))).toEqual([]);
    expect(
        Object.keys(metafile.inputs).some((path) => /(?:playwright|bun-webgpu)/.test(path)),
    ).toBe(false);
});

test("Bun engine exports load without Node-only modules or project tooling", async () => {
    const result = await Bun.build({
        entrypoints: [resolve(ROOT, "scripts/bun-engine-entry.ts")],
        target: "bun",
        outdir: resolve(ROOT, ".artifacts/entry-bun"),
        metafile: true,
    });
    const metafile = checkBuild(result, "Bun engine");
    expect(
        edgesOf(metafile).filter((path) => /^(?:node:|fs$|path$|os$|child_process$)/.test(path)),
    ).toEqual([]);
    expect(Object.keys(metafile.inputs).some((path) => path.includes("src/project/"))).toBe(false);
});

test("the Node Vite entry stays outside the browser and Bun engine graph", async () => {
    const result = await Bun.build({
        entrypoints: [resolve(ROOT, "scripts/node-vite-entry.ts")],
        target: "node",
        external: [
            "vite",
            "unplugin-typegpu",
            "unplugin-typegpu/vite",
            "node:fs",
            "node:path",
            "fs",
            "path",
        ],
        outdir: resolve(ROOT, ".artifacts/entry-node"),
        metafile: true,
    });
    const metafile = checkBuild(result, "Node Vite");
    expect(edgesOf(metafile).filter((path) => path.startsWith("bun:"))).toEqual([]);
    expect(
        Object.keys(metafile.inputs).filter((path) =>
            /src\/(?:engine|core|standard|extras|transitional)\//.test(path),
        ),
    ).toEqual([]);
});
