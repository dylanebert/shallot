import type { Plugin } from "@dylanebert/shallot";

export interface DiscoveredPlugin {
    name: string;
    plugin: Plugin;
}

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
