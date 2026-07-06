import { type Page, test } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;
const SCENE = "scenes/scene.scene";

// the headline external-change round-trip (testing.md "Reload tier"). The conflict DECISION is bun-tested
// (project/index.test.ts classifyExternal); this drives the live wiring those tests can't reach — the dev
// server's watcher → `shallot:external` event → the editor's reload-vs-conflict handling — against the real
// editor in FILE mode (not the capture default ?save=off, which never writes). A clean external edit
// reloads the editor; an external edit made while the editor has unsaved changes raises a conflict instead
// of silently nuking them, and Keep mine writes the local version back losslessly. The external writes go
// through /__api/file (a writer the editor's own `recentSaves` dedup doesn't cover, so the watcher treats
// it as external), and the fixture scene is restored on the way out.

async function readScene(page: Page): Promise<string> {
    return (await page.request.get(`${API}/file?path=${SCENE}`)).text();
}

async function writeScene(page: Page, content: string): Promise<void> {
    await page.request.post(`${API}/file?path=${SCENE}`, { data: { content } });
}

// the editor's live serialization, read through the control seam; null while the ws is down (mid-reload)
async function editorScene(page: Page): Promise<string | null> {
    const res = await page.request.get(`${API}/state`);
    if (!res.ok()) return null;
    return ((await res.json()) as { scene?: string }).scene ?? null;
}

async function pollScene(
    page: Page,
    want: (s: string) => boolean,
    timeoutMs = 12_000,
): Promise<string | null> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const s = await editorScene(page);
        if (s !== null && want(s)) return s;
        await page.waitForTimeout(150);
    }
    return null;
}

async function mount(page: Page): Promise<void> {
    await page.waitForSelector(S.editor, { timeout: 6000 });
    await page.waitForSelector(S.canvas, { timeout: 3000 });
    await page.waitForFunction(
        () => {
            const c = document.querySelector("canvas");
            return c != null && c.width > 0 && c.height > 0;
        },
        null,
        { timeout: 6000 },
    );
    await page.waitForTimeout(300);
}

defineFlow(
    { name: "external-change", scene: "scene", target: "manual" },
    async ({ page, step, act, assert }) => {
        test.setTimeout(60_000);

        // file mode (no ?save=off): the editor reads from + writes to the fixture scene. Restored in finally.
        await page.goto(`http://localhost:${PORT}`);
        await mount(page);

        const original = await readScene(page);
        assert("fixture scene loaded with the box entity", original.includes('id="box"'));

        try {
            // 1 — clean external edit: the editor has nothing unsaved, so an on-disk change reloads it
            await writeScene(page, original.replace('id="box"', 'id="boxer"'));
            const reloaded = await pollScene(page, (s) => s.includes('id="boxer"'));
            assert(
                "a clean external edit reloaded the editor with the new content",
                reloaded !== null,
            );
            await step("clean-reload", { highlight: [S.viewport] });

            // 2 — conflict: an unsaved edit in the editor, then an external edit to the same scene. The
            // pending autosave is cancelled and a conflict raised — never a silent reload that drops the edit
            await page.request.post(`${API}/command`, {
                data: { method: "setId", args: { id: "boxer", newId: "localbox" } },
            });
            assert(
                "the editor holds the unsaved rename",
                (await editorScene(page))?.includes('id="localbox"') === true,
            );
            await writeScene(page, reloaded!.replace('id="ground"', 'id="grounded"'));

            const conflict = page.locator(S.banner, { hasText: "changed on disk" });
            await conflict.waitFor({ timeout: 5000 });
            assert(
                "an external edit mid-edit raised a conflict, not a silent reload",
                await conflict.isVisible(),
            );

            const midDisk = await readScene(page);
            assert(
                "the autosave was cancelled — the external version is intact on disk",
                midDisk.includes('id="grounded"'),
            );
            assert(
                "the editor did not clobber the external edit with the unsaved local one",
                !midDisk.includes('id="localbox"'),
            );
            await step("conflict", { highlight: [S.banner] });

            // Keep mine resolves it local-wins: the editor's version is written back, losslessly
            await act.click(page.locator(`${S.banner} .action`, { hasText: "Keep mine" }));
            await act.wait(400);

            const resolved = await readScene(page);
            const editor = await editorScene(page);
            assert(
                "Keep mine wrote the local version over the external one",
                resolved.includes('id="localbox"') && !resolved.includes('id="grounded"'),
            );
            assert(
                "the written file is byte-identical to the editor's serialization (lossless)",
                resolved === editor,
            );
            assert(
                "the conflict banner cleared once resolved",
                (await page.locator(S.banner, { hasText: "changed on disk" }).count()) === 0,
            );
            await step("resolved", { highlight: [S.viewport] });
        } finally {
            await writeScene(page, original);
        }
    },
);
