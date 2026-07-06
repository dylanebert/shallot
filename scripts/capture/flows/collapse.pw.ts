import { defineFlow } from "../runner";
import { S } from "../selectors";

// the sidebar collapse interaction (App.svelte's ⌘\ / Ctrl+\ shortcut → setCollapsed). Baseline net for
// Phase 0 — a real keypress toggles both panels. The routing math is unit-tested (keymap.test.ts); this
// asserts the wiring. Selection is cleared first so the inspector auto-expand effect doesn't fight it.
const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

defineFlow({ name: "collapse", scene: "demo" }, async ({ page, step, act, assert }) => {
    await page.request.post(`${API}/command`, { data: { method: "clearSelection" } });
    await act.wait(200);

    assert("sidebars start expanded", (await page.locator(".sidebar.collapsed").count()) === 0);
    await step("expanded", { highlight: [S.outliner, S.inspector] });

    await act.key("Control+\\");
    assert("both sidebars collapse", (await page.locator(".sidebar.collapsed").count()) === 2);
    await step("collapsed");

    await act.key("Control+\\");
    assert("both sidebars expand again", (await page.locator(".sidebar.collapsed").count()) === 0);
});
