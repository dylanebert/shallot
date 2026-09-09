import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { plan } from "../src/project/generate";
import { resolveLocalModules } from "../src/project/host";
import { normalize } from "../src/project/manifest";
import { manifestPath } from "../src/project/vite";
import { installGpuGlobals } from "./gpu-globals";

/** System-webview gaps beyond the base floor; null means the base floor itself is unavailable.
 * WKWebView's Safari 26.5 / Apple Silicon audit met the floor including timestamp-query;
 * subgroups is preferred (LDS fallback). WebView2 is Chromium; WebKitGTK has no usable WebGPU.
 */
export const WEBVIEW_UNSUPPORTED: Record<string, readonly string[] | null> = {
    mac: [],
    windows: [],
    linux: null,
};

function readManifest(absDir: string) {
    try {
        return normalize(readFileSync(manifestPath(absDir), "utf-8"));
    } catch {
        return {};
    }
}

/**
 * the WebGPU features a project's enabled plugins require beyond the base floor — the same union the
 * runtime computes at `build()` (engine/app: `plugins.flatMap(p => p.features)`), resolved statically
 * from `shallot.json`. Imports the engine barrel under the GPU-constants shim, since the barrel
 * evaluates GPU module code at import (sear's top-level `GPUShaderStage`); local plugins import from
 * their project-resolved identity. Missing entries refuse before evaluation; an already-resolved
 * local that fails during evaluation remains the web build's problem to surface.
 */
export async function requiredFeatures(projectDir: string): Promise<string[]> {
    installGpuGlobals(); // install GPUShaderStage etc. so the barrel + locals import under the plain `bun` CLI
    const absDir = resolve(projectDir);
    const project = plan(readManifest(absDir), absDir);
    const locals = resolveLocalModules({ dir: absDir, locals: project.locals });
    const { engine } = project;

    const shallot = (await import("@dylanebert/shallot")) as unknown as Record<
        string,
        { features?: readonly string[] } | undefined
    >;
    const features = new Set<string>();

    for (const name of engine) {
        for (const f of shallot[`${name}Plugin`]?.features ?? []) features.add(f);
    }
    for (const local of locals) {
        try {
            const mod = (await import(local.path)) as {
                default?: { features?: readonly string[] };
            };
            for (const f of mod.default?.features ?? []) features.add(f);
        } catch {}
    }
    return [...features];
}

/** Refusal lines for a backend missing the base floor or required plugin features; preferred
 * features never enter `required`. Empty means allowed. Portable CEF supplies its own Chromium. */
export function verdict(target: string, portable: boolean, required: readonly string[]): string[] {
    if (portable) return [];
    const unsupported = WEBVIEW_UNSUPPORTED[target];
    const missing =
        unsupported === null
            ? ["WebGPU base floor"]
            : (unsupported ?? []).filter((f) => required.includes(f));
    if (missing.length === 0) return [];
    return [
        `Cannot build ${target}: the system webview lacks required ${missing.join(", ")}.`,
        "Rebuild with --portable for the bundled Chromium runtime.",
    ];
}

/** Refuse before native emission or launch when the selected backend cannot run the project. */
export async function requireBackend(projectDir: string, target: string, portable: boolean) {
    const lines = verdict(target, portable, await requiredFeatures(projectDir));
    if (lines.length === 0) return;
    for (const line of lines) console.error(`  ${line}`);
    process.exit(1);
}
