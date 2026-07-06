import { defineFlow } from "../runner";
import { S } from "../selectors";

// the outliner `+` summons the Add menu (Empty first, then primitives grouped by category). Picking a
// bundle drops a ready-to-use entity. This flow drives the Empty (bare-entity) path — via keyboard, so
// the menu's nav + key containment are exercised; `add-primitive` covers the visible-primitive path.
defineFlow({ name: "add-entity", scene: "demo" }, async ({ page, step, act, assert }) => {
    const beforeCount = await page.locator(S.row).count();

    // select a row first so the Delete-containment assert below has something a leak would delete
    await act.click(page.locator(S.row, { hasText: "box" }).first());

    await step("before", { highlight: [S.addEntityBtn] });
    await act.click(S.addEntityBtn);
    await act.waitFor(S.addEntityMenu);

    const first = (await page.locator(S.addEntityItem).first().textContent())?.trim();
    assert("Empty leads the menu", first === "Empty");
    await step("menu-open", { highlight: [S.addEntityMenu] });

    // the summoned menu owns the keyboard: Delete must not reach the window keymap and delete the
    // selection underneath
    await act.key("Delete");
    assert(
        "Delete inside the menu is contained",
        (await page.locator(S.row).count()) === beforeCount,
    );
    assert("menu stays open", (await page.locator(S.addEntityMenu).count()) === 1);

    // keyboard pick: ArrowDown focuses Empty (first item), Enter drops it
    await act.key("ArrowDown");
    await act.key("Enter");
    assert("menu closed on pick", (await page.locator(S.addEntityMenu).count()) === 0);

    const afterCount = await page.locator(S.row).count();
    assert("new entity appeared", afterCount > beforeCount);

    await step("after", { highlight: [S.outliner] });
});
