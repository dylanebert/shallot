import { defineFlow } from "../runner";
import { S } from "../selectors";

// The viewport-pick wiring gate (testing.md "Editor tiers": "a viewport click selects and matches the
// outliner"). A real left-click on the canvas — not the control-API select the `verify` flow drives —
// reads the entity under the cursor from the prepass id lane (App.svelte → lib/pick) and routes it to
// doc.select, the same path the outliner uses. Center hits the box/ground (the `verify` flow confirms the
// center renders geometry); a top corner is empty sky, so it clears the selection.
//
// `locator.hover()` is the move that updates the engine's `Inputs.mouse` (a bare `page.mouse.move` from
// the test's start position does not), and the pick readback samples the cursor there. Hover, let the
// 1-2 frame + staging-ring readback resolve `Pick.eid` (intentionally stale-but-cheap, like orrstead's
// hover — fine for a still tap), then press + release in place.
defineFlow({ name: "pick-viewport", scene: "demo" }, async ({ page, step, assert, act }) => {
    const canvas = page.locator(S.canvas);

    // center: hover (so the readback resolves the entity there), then click
    await canvas.hover();
    await act.wait(250);
    await page.mouse.down();
    await page.mouse.up();
    await act.wait(300);

    await step("picked", { highlight: [S.outliner], click: S.canvas });

    const selected = page.locator(S.rowSelected);
    assert("a viewport click selects one entity", (await selected.count()) === 1);
    const name = (await selected.innerText()).trim();
    assert(`the selection is a scene entity (got "${name}")`, /box|ground/.test(name));

    // a top corner is empty sky — clicking there clears the selection
    await canvas.hover({ position: { x: 8, y: 8 } });
    await act.wait(250);
    await page.mouse.down();
    await page.mouse.up();
    await act.wait(300);
    assert("clicking empty space deselects", (await page.locator(S.rowSelected).count()) === 0);
});
