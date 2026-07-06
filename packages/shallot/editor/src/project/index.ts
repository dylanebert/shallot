export async function fetchScene(dir: string, path: string): Promise<string> {
    const res = await fetch(
        `/__api/scene?dir=${encodeURIComponent(dir)}&path=${encodeURIComponent(path)}`,
    );
    if (!res.ok) throw new Error("Failed to fetch scene");
    return res.text();
}

export async function saveScene(dir: string, path: string, content: string): Promise<void> {
    const res = await fetch(
        `/__api/scene?dir=${encodeURIComponent(dir)}&path=${encodeURIComponent(path)}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content }),
        },
    );
    if (!res.ok) throw new Error("Failed to save scene");
}

export async function fetchManifest(dir: string): Promise<string> {
    const res = await fetch(`/__api/manifest?dir=${encodeURIComponent(dir)}`);
    if (!res.ok) throw new Error("Failed to fetch manifest");
    return res.text();
}

export async function saveManifest(dir: string, content: string): Promise<void> {
    const res = await fetch(`/__api/manifest?dir=${encodeURIComponent(dir)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
    });
    if (!res.ok) throw new Error("Failed to save manifest");
}

export type SaveMode = "file" | "ephemeral";

// `?save=off` opens the editor in ephemeral mode — a full live view that never writes. The capture
// harness and engine-dev sessions use it so testing the editor doesn't dirty the working tree.
export function pickSaveMode(search: string): SaveMode {
    return new URLSearchParams(search).get("save") === "off" ? "ephemeral" : "file";
}

// The single project I/O boundary — scenes and the `shallot.json` manifest. Loads always read (ephemeral
// is read-only, not blind); writes no-op in ephemeral mode (the capture harness / engine-dev `?save=off`
// sessions never dirty the tree). A future web build swaps a server backend in here without touching the editor.
export const Persist = {
    mode: "file" as SaveMode,
    load: fetchScene,
    async save(dir: string, path: string, content: string): Promise<void> {
        if (Persist.mode === "ephemeral") return;
        await saveScene(dir, path, content);
    },
    loadManifest: fetchManifest,
    async saveManifest(dir: string, content: string): Promise<void> {
        if (Persist.mode === "ephemeral") return;
        await saveManifest(dir, content);
    },
};

export type ExternalChange = "ignore" | "reload" | "conflict";

// classify a file changing on disk outside the editor (an IDE save, a git checkout, dev mode). The dev
// server reports the project-relative path and whether it's the manifest — its watcher already narrowed
// to scenes + `shallot.json`, and `recentSaves` already dropped the editor's own write. A change to the
// active scene or the manifest reloads when nothing is unsaved and becomes a conflict the author resolves
// when something is — never the silent reload that would discard unsaved edits + session state. A change
// to any other scene can't touch the active content, so it's ignored.
export function classifyExternal(change: {
    path: string;
    manifest: boolean;
    scene: string | null;
    unsaved: boolean;
}): ExternalChange {
    const relevant = change.manifest || change.path === change.scene;
    if (!relevant) return "ignore";
    return change.unsaved ? "conflict" : "reload";
}

// match a config `scene` (public-relative, e.g. "scenes/x.scene") against the discovered scene list
// (project-relative, e.g. "public/scenes/x.scene"). the two frames differ by the public dir, so
// compare on a trailing path-segment boundary. an array config picks its first scene.
export function matchScene(want: string | string[] | undefined, scenes: string[]): string | null {
    const path = (Array.isArray(want) ? want[0] : want)?.replace(/^\.?\//, "");
    if (!path) return null;
    return scenes.find((s) => s === path || s.endsWith(`/${path}`)) ?? null;
}
