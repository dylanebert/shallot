import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_REACT_PORT || "3008";
const API = `http://localhost:${PORT}/__api`;
const FILE = "src/game.ts";

// The React leg of Stage 1: a STANDARD vite+react project opened in the editor — proving a user can build
// their game UI in React and still edit the project in the editor. @vitejs/plugin-react has no name
// collision with the editor's svelte plugin, so it runs in the merged pipeline and transforms the project's
// .tsx (JSX); react + react-dom resolve as project deps so the HUD renders. Same control-seam proof for
// alias + bare dep, plus the mounted React HUD, then re-checked across an HMR edit.

interface GameRead {
    alias: number;
    dep: number;
}

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
    { name: "framework-react", scene: "scene", target: "manual", timeout: 60_000 },
    async ({ page, step, assert, openEditor }) => {
        test.setTimeout(60_000);
        await openEditor(Number(PORT));

        // the project plugin compiled + ran: `alias` via the project's resolve.alias, `dep` via the bare
        // project dep wgpu-matrix. Both are 7 only because the host merged the project's vite.config.
        const ready = await poll(page, (g) => g != null && g.alias === 7 && g.dep === 7);
        assert("project alias + bare dep resolved through the merged config", ready);

        // the .tsx HUD compiled (the project's own @vitejs/plugin-react transformed the JSX) and rendered
        // (react + react-dom resolved as project deps) over the canvas — the full React toolchain working
        // through shallot-as-tool, a React game UI mounting in the editor.
        const hud = page.locator(".game-hud[data-framework='react']");
        await hud.waitFor({ state: "attached", timeout: 5000 });
        const text = (await hud.textContent())?.trim();
        assert("a React component compiled + rendered in the editor", text === "react");
        await step("loaded", { highlight: [S.viewport] });

        // the merge survives HMR: edit the plugin through the dev server's /__api/file seam, bumping the
        // system's output; the merged dev server recompiles + swaps it live.
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
