import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

// Toggling a plugin in the editor writes the project's shallot.json — the single enablement truth dev mode
// also reads (Stage 1's headline: editor and dev mode load from one declaration). The capture harness runs
// the fixture ephemeral (?save=off, no writes), so this flow opts into file mode, drives the real menu
// toggle, asserts the manifest changed on disk + the editor cleanly rebuilds, then restores the manifest so
// the fixture stays clean. The write LOGIC is unit-tested (manifest.test.ts); this proves the wiring:
// menu click → manifest write → reactive rebuild (no full reload — recentSaves dedups the editor's own write).
defineFlow(
    { name: "plugin-toggle", scene: "demo" },
    async ({ page, step, act, assert, sample }) => {
        // re-open in file mode: the runner navigated ?save=off, but the toggle must actually write to be asserted
        await page.goto(`http://localhost:${PORT}`);
        await page.waitForSelector(S.editor);
        await page.waitForFunction(() => {
            const c = document.querySelector("canvas");
            return !!c && c.clientWidth > 0 && c.clientHeight > 0;
        });
        await act.wait(500);

        const read = async () => (await page.request.get(`${API}/manifest`)).text();
        const original = await read();
        assert("fixture manifest has Profile enabled", original.includes('"Profile"'));
        await step("before", { highlight: [S.menuBtn] });

        // open the plugin menu and toggle Profile (an enabled extra) off
        await act.click(S.menuBtn);
        await act.waitFor(S.menuDropdown);
        await act.click(S.menuPlugins);
        await act.waitFor(S.menuSubmenu);
        const profile = page.locator(S.menuRow, { hasText: "Profile" });
        assert("Profile listed in the plugin menu", await profile.isVisible());
        await profile.click();

        // the editor wrote shallot.json; poll the disk until the toggle lands (Profile removed)
        let after = original;
        for (let i = 0; i < 30 && after === original; i++) {
            await act.wait(150);
            after = await read();
        }
        assert(
            "manifest changed on disk — Profile removed",
            after !== original && !after.includes('"Profile"'),
        );

        // the write rebuilds the State reactively (recentSaves dedups it, so no full reload) — render survives
        await act.wait(800);
        const rendered = await sample();
        assert(
            `editor rebuilt and renders after the toggle (non-background ${rendered.nonBackground.toFixed(2)})`,
            rendered.nonBackground > 0.3,
        );
        await step("toggled");

        // restore the fixture manifest so the working tree stays clean
        await page.request.post(`${API}/manifest`, { data: { content: original } });
    },
);
