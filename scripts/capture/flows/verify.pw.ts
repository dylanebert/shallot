import { defineFlow } from "../runner";
import { S } from "../selectors";

// The editor behavioral gate (testing.md "Editor tiers"): two bands in one session. DOM-state asserts the
// chrome reflects the document; render asserts the engine actually draws into the viewport in edit mode —
// the contract that fails a boot-clean-but-blank editor. The fixture (scripts/capture.ts) carries
// ProfilePlugin and the named `ground` / `box` entities. Setup + mutation ride the shallot:request control
// API (the `/__api/*` endpoints, the same seam an agent drives); selection too, so the DOM reflection is
// the assertion. Screenshots are a side artifact of the same flow.
const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

// the edit-mode selection outline (App.svelte sets Outline.color = 0xff6a00); its near-zero blue channel
// separates it from the lit box surface (warm orange, high blue) and the grey ground.
const OUTLINE: [number, number, number] = [255, 106, 0];

defineFlow({ name: "verify", scene: "demo" }, async ({ page, step, act, assert, sample }) => {
    // ---- Band 1: DOM-state ----

    // the fixture's config enables ProfilePlugin, but the editor must not float the engine HUD over its
    // chrome — the overlay is F3-toggled and lives within the viewport, so it's absent by default. Red
    // before the fix: ProfileRenderSystem auto-mounted the overlay to document.body on the first frame.
    assert(
        "profiler overlay absent by default",
        (await page.locator("[data-shallot-profile]").count()) === 0,
    );

    // F3 mounts the overlay inside the canvas container (not floated over the chrome); F3 again removes it
    await act.key("F3");
    assert(
        "F3 mounts the profiler overlay within the canvas",
        (await page.locator(`${S.viewportCanvas} [data-shallot-profile]`).count()) === 1,
    );
    await act.key("F3");
    assert(
        "F3 again removes the profiler overlay",
        (await page.locator("[data-shallot-profile]").count()) === 0,
    );

    // node titles read as id or type, never the bare "entity" fallback (S.row is already outliner-scoped)
    const labels = (await page.locator(`${S.row} ${S.label}`).allInnerTexts()).map((t) => t.trim());
    assert("outliner lists the scene", labels.length >= 4);
    assert(
        "no node titles as 'entity'",
        labels.every((t) => t !== "entity"),
    );
    assert(
        "named entities title by id",
        labels.some((t) => t.includes("ground")) && labels.some((t) => t.includes("box")),
    );

    // read the document through the control API — the seam an agent uses, no test backdoor
    const state = await (await page.request.get(`${API}/state`)).json();
    assert(
        "control API returns the scene",
        typeof state.scene === "string" && state.scene.includes("ground"),
    );

    // drive selection through the control API and assert the DOM reflects it
    const sel = await (
        await page.request.post(`${API}/command`, {
            data: { method: "select", args: { ids: ["ground"] } },
        })
    ).json();
    assert("select command applied", sel.ok === true);
    await act.wait(200);

    assert("one row carries .selected", (await page.locator(S.rowSelected).count()) === 1);
    assert(
        "the selected row is 'ground'",
        (await page.locator(S.rowSelected).innerText()).includes("ground"),
    );
    assert(
        "inspector renders sections for the selection",
        (await page.locator(`${S.inspector} ${S.sectionHeader}`).count()) > 0,
    );

    // the control API observes the same selection
    const after = await (await page.request.get(`${API}/state`)).json();
    assert("control API reports the selection", after.selection.includes("ground"));

    await step("dom-state", { highlight: [S.outliner, S.inspector] });

    // ---- Band 2: render (pixels) ----

    // clear selection so no outline is painted, then assert the viewport is not blank: the box + ground
    // fill the center. Red before the slab fix — with SlabSystem skipped in edit mode the transform
    // firehose never uploads, so geometry renders nowhere and the center stays the clear color.
    await page.request.post(`${API}/command`, { data: { method: "clearSelection" } });
    await act.wait(300);

    const blank = await sample({ accent: OUTLINE });
    assert(
        `edit-mode viewport renders geometry (non-background ${blank.nonBackground.toFixed(2)})`,
        blank.nonBackground > 0.3,
    );
    assert(
        `no outline without a selection (accent ${blank.accent.toFixed(4)})`,
        blank.accent < 0.001,
    );

    // select the box through the control API; the outline pass paints an accent edge around it
    await page.request.post(`${API}/command`, {
        data: { method: "select", args: { ids: ["box"] } },
    });
    await act.wait(300);

    await step("render", { highlight: [S.viewport] });

    const outlined = await sample({ region: { x: 0.2, y: 0.2, w: 0.6, h: 0.6 }, accent: OUTLINE });
    assert(
        `selection paints outline pixels near the box (accent ${outlined.accent.toFixed(4)})`,
        outlined.accent > 0.002,
    );
});
