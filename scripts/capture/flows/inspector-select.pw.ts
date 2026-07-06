import { defineFlow } from "../runner";
import { S } from "../selectors";

// The enum-field dropdown wiring gate (testing.md "Editor tiers": the flow asserts behavior, the
// screenshot is the side artifact). Inspector enum fields render a custom Select — a themed,
// `fit`-positioned menu replacing the native `<select>`, whose popup opens at the OS / native-webview
// layer outside the editor's on-screen guard. Drive it through the keyboard contract (`lib/select.ts`):
// open, confirm the menu is on-screen, arrow to the next option, commit, assert the value changed.
// Keyboard, not pointer — the orbit field sits deep in a scrolling panel where the headless hit-test is
// unreliable for any nested field; the keyboard path is the one the bug report's "input logic" names.
defineFlow({ name: "inspector-select", scene: "demo" }, async ({ page, step, assert, act }) => {
    await act.click(page.locator(S.row, { hasText: "camera" }));

    // scope to the Orbit section — Camera also carries a `mode` enum, so the bare field name is ambiguous
    const trigger = page
        .locator(`${S.inspector} .section:has(.section-label:text-is('orbit'))`)
        .locator(`${S.fieldRow}:has(${S.fieldLabel}:text-is('mode'))`)
        .locator(S.selectTrigger);
    assert("orbit mode renders a Select trigger", (await trigger.count()) === 1);
    assert("the trigger shows the default value", (await trigger.innerText()).trim() === "free");

    await trigger.focus();
    await act.key("Enter");

    const menu = page.locator(S.selectMenu);
    assert("Enter opens the listbox", await menu.isVisible());
    assert(
        "the menu lists both OrbitMode options",
        (await page.locator(S.selectOption).count()) === 2,
    );

    // the fix the bug report names: the menu is pinned on-screen by `fit`, never off the viewport edge.
    const box = await menu.boundingBox();
    const vp = page.viewportSize()!;
    assert(
        `the menu fits the viewport (got ${JSON.stringify(box)})`,
        !!box &&
            box.x >= 0 &&
            box.y >= 0 &&
            box.x + box.width <= vp.width &&
            box.y + box.height <= vp.height,
    );

    await step("mode-open", { highlight: [S.selectMenu], clip: S.inspector });

    // arrow to the next option (free → locked) and commit
    await act.key("ArrowDown");
    await act.key("Enter");

    assert("Enter closes the menu after a pick", (await menu.count()) === 0);
    assert("the trigger reflects the new value", (await trigger.innerText()).trim() === "locked");
});
