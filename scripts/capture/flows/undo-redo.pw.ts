import { defineFlow } from "../runner";
import { S } from "../selectors";

// the undo/redo interaction baseline: add an entity (a real outliner gesture), then drive ⌘Z / ⌘⇧Z / ⌘Y
// through real key events and assert the document tree reverts and replays. The command dispatch is
// unit-tested (commands.test.ts, editor.test.ts); this is the keyboard wiring over the live editor —
// the shift chord matters here because a real shifted keydown reports e.key "Z", not "z".
defineFlow({ name: "undo-redo", scene: "demo" }, async ({ page, step, act, assert }) => {
    const before = await page.locator(S.row).count();

    await act.click(S.addEntityBtn);
    await act.click(page.locator(S.addEntityItem, { hasText: "Empty" }));
    const added = await page.locator(S.row).count();
    assert("add entity grows the outliner", added === before + 1);
    await step("added", { highlight: [S.outliner] });

    await act.key("Control+z");
    assert("undo removes the added entity", (await page.locator(S.row).count()) === before);
    await step("undone", { highlight: [S.outliner] });

    await act.key("Control+y");
    assert("redo restores it", (await page.locator(S.row).count()) === before + 1);

    await act.key("Control+z");
    assert("undo again removes it", (await page.locator(S.row).count()) === before);
    await act.key("Control+Shift+z");
    assert("ctrl+shift+z also redoes", (await page.locator(S.row).count()) === before + 1);
});
