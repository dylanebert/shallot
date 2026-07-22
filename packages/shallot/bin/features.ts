import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { plan } from "../src/project/generate";
import { normalize } from "../src/project/manifest";
import { manifestPath } from "../src/project/vite";
import { installGpuGlobals } from "./gpu-globals";

// Required features (beyond the base floor) a wry/system-webview backend can't provide, keyed by
// target — the hard "won't run" gaps the build warns about. `subgroups` is NOT one: physics lists it
// as a `preferredFeatures` (LDS fallback on WKWebView), so it's never in the required set and a macOS
// system-webview physics build runs. Windows WebView2 is full Chromium. Linux WebKitGTK has no usable
// WebGPU at all — a different shape, handled directly in verdict(). Every CEF (`--portable`) build
// ships its own Chromium, so it never appears here. No required gap stands today; the map is the seam
// for the next required feature beyond the floor. See gpu.ts BASE_FEATURES + Plugin.features.
const WEBVIEW_UNSUPPORTED: Record<string, readonly string[]> = {
    mac: [],
    windows: [],
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
 * their specifier the same way. A local that fails to import is the web build's problem to surface.
 */
export async function requiredFeatures(projectDir: string): Promise<string[]> {
    installGpuGlobals(); // install GPUShaderStage etc. so the barrel + locals import under the plain `bun` CLI
    const absDir = resolve(projectDir);
    const { engine, locals } = plan(readManifest(absDir), absDir);

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

/**
 * build-time warning lines for a (target, portable) backend given the project's required features —
 * empty when the chosen backend can render the app. Never blocks: a warned build still produces an
 * artifact that reaches the engine's diagnostic tier at launch, matching the loud-boundary contract.
 */
export function verdict(target: string, portable: boolean, required: readonly string[]): string[] {
    if (portable) return []; // CEF ships its own Chromium — every feature, every platform
    if (target === "linux") {
        return [
            "linux default uses WebKitGTK, which has no usable WebGPU — the app reaches the diagnostic",
            "tier at launch. Rebuild with --portable for the bundled Chromium runtime.",
        ];
    }
    const missing = (WEBVIEW_UNSUPPORTED[target] ?? []).filter((f) => required.includes(f));
    if (missing.length === 0) return [];
    return [
        `${target} default uses the system webview, which lacks: ${missing.join(", ")} (required by`,
        "the project's plugins). Rebuild with --portable for the bundled Chromium runtime.",
    ];
}
