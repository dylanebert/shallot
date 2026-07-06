import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_SVELTE_PORT || "3007";
const API = `http://localhost:${PORT}/__api`;
const FILE = "src/game.ts";

// Stage 1 of the tool-first editor track: a STANDARD vite+svelte project (its own vite.config) opened in
// the editor. The host merging the project's config (bin/edit.ts) is what makes three things resolve, none
// of which the editor-first path could: resolve.alias ($game), a bare project dep (wgpu-matrix), and the
// project's svelte plugin (deduped by name against the editor's, so one runes instance compiles the
// .svelte HUD). Asserted positively through the control seam + the mounted HUD, then re-checked across an
// HMR edit so the merged config survives the live-reload path too.

interface GameRead {
    alias: number;
    dep: number;
}

// the warm-spawned Game entity, read through the control seam (a system keeps `dep` written). null while
// the ws is mid-rebuild or the engine isn't up yet — the caller polls.
async function game(page: Page): Promise<GameRead | null> {
    const res = await page.request.get(`${API}/entities?component=Game`);
    if (!res.ok()) return null;
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) return null;
    const ents = data as { components: { game?: GameRead } }[];
    return ents[0]?.components?.game ?? null;
}

async function poll(
    page: Page,
    want: (g: GameRead | null) => boolean,
    timeoutMs = 10_000,
): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (want(await game(page))) return true;
        await page.waitForTimeout(150);
    }
    return false;
}

defineFlow(
    { name: "framework-svelte", scene: "scene", target: "manual", timeout: 60_000 },
    async ({ page, step, assert, openEditor }) => {
        test.setTimeout(60_000);
        await openEditor(Number(PORT));

        // the project plugin compiled + ran: `alias` resolved via the project's resolve.alias, `dep` via
        // the bare project dep wgpu-matrix. Both are 7 only because the host merged the project's vite.config.
        const ready = await poll(page, (g) => g != null && g.alias === 7 && g.dep === 7);
        assert("project alias + bare dep resolved through the merged config", ready);

        // a project .svelte component compiled (the editor's svelte over the project's component, its own
        // svelte plugin deduped by name) and mounted over the canvas — a svelte game UI in the editor.
        const hud = page.locator(".game-hud[data-framework='svelte']");
        await hud.waitFor({ state: "attached", timeout: 5000 });
        const text = (await hud.textContent())?.trim();
        assert("a svelte component compiled + mounted in the editor", text === "svelte");
        await step("loaded", { highlight: [S.viewport] });

        // the merge survives HMR: edit the plugin through the dev server's /__api/file seam (where the
        // watcher watches), bumping the system's output; the merged dev server recompiles + swaps it live.
        const original = await (await page.request.get(`${API}/file?path=${FILE}`)).text();
        try {
            await page.request.post(`${API}/file?path=${FILE}`, {
                data: { content: original.replace("const BUMP = 0;", "const BUMP = 1;") },
            });
            const bumped = await poll(page, (g) => g != null && g.dep === 8);
            assert("an HMR edit recompiled the project plugin through the merged config", bumped);
            await step("hmr", { highlight: [S.viewport] });
        } finally {
            await page.request.post(`${API}/file?path=${FILE}`, { data: { content: original } });
        }
    },
);
