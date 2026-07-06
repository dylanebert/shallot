import type { Page } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

// the GPU half of the reload-conformance sweep (testing.md "Reload tier"): the bun-test harness
// (tests/conformance.test.ts) runs the rebuild loop per plugin on the bun-webgpu adapter; the
// default GPU plugin set (slab, transforms, render, part, sear, glaze) gets the same loop here,
// live in the editor — play/stop toggled twice and a scene switched away and back, each rebuild
// re-running every lifecycle against the same module singletons. Idempotence is the assertion:
// entity counts never drift (a doubling warm spawn or uncleaned registry shows as a count or a
// dead canvas), and the viewport still renders after every rebuild. The real-HMR-swap half of the
// live loop is the `hot-reload` flow — not repeated here.

async function count(page: Page, component: string): Promise<number> {
    const res = await page.request.get(`${API}/entities?component=${component}`);
    return ((await res.json()) as unknown[]).length;
}

defineFlow(
    { name: "reload-conformance", scene: "demo" },
    async ({ page, step, act, assert, sample }) => {
        await page.waitForTimeout(500);
        const parts = await count(page, "part");
        const tickers = await count(page, "ticker");
        assert(`baseline: demo scene parts present (got ${parts})`, parts === 2);
        assert(`baseline: one warm-derived ticker (got ${tickers})`, tickers === 1);
        const base = await sample();
        assert(
            `baseline: viewport renders (non-background ${base.nonBackground.toFixed(2)})`,
            base.nonBackground > 0.3,
        );

        // two full play/stop cycles — each play builds a fresh play State, each stop returns to the
        // edit State, all against the same module singletons
        for (let cycle = 1; cycle <= 2; cycle++) {
            await act.click(S.playBtn);
            await act.wait(800);
            const playParts = await count(page, "part");
            const playTickers = await count(page, "ticker");
            assert(`play ${cycle}: parts stable (got ${playParts})`, playParts === parts);
            assert(
                `play ${cycle}: ticker re-derived once, never doubled (got ${playTickers})`,
                playTickers === 1,
            );
            const playing = await sample();
            assert(
                `play ${cycle}: viewport renders (non-background ${playing.nonBackground.toFixed(2)})`,
                playing.nonBackground > 0.3,
            );
            await act.click(S.stopBtn);
            await act.wait(500);
        }
        const editParts = await count(page, "part");
        const editTickers = await count(page, "ticker");
        assert(`after play/stop ×2: parts stable (got ${editParts})`, editParts === parts);
        assert(`after play/stop ×2: one ticker (got ${editTickers})`, editTickers === 1);
        await step("after-play-cycles", { highlight: [S.viewport] });

        // scene switch away and back — the File submenu lists both fixture scenes; alt carries one part so
        // the counts tell the scenes apart
        const switchTo = async (label: string) => {
            await act.click(S.menuBtn);
            await act.waitFor(S.menuDropdown);
            await act.click(S.menuFile);
            await act.waitFor(S.menuSubmenu);
            await act.click(page.locator(S.menuRow, { hasText: label }).last());
            await act.wait(800);
        };
        await switchTo("alt");
        const altParts = await count(page, "part");
        assert(`alt scene loaded (parts ${altParts})`, altParts === 1);
        const alt = await sample();
        assert(
            `alt scene renders (non-background ${alt.nonBackground.toFixed(2)})`,
            alt.nonBackground > 0.1,
        );
        await step("alt-scene", { highlight: [S.viewport] });

        await switchTo("scene");
        const backParts = await count(page, "part");
        const backTickers = await count(page, "ticker");
        assert(`demo scene restored (parts ${backParts})`, backParts === parts);
        assert(`demo restore: one ticker, never doubled (got ${backTickers})`, backTickers === 1);
        const restored = await sample();
        assert(
            `restored scene renders (non-background ${restored.nonBackground.toFixed(2)})`,
            restored.nonBackground > 0.3,
        );
        await step("restored", { highlight: [S.viewport] });
    },
);
