import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "app-menu", scene: "demo" }, async ({ page, step, act, assert }) => {
    await act.click(S.menuBtn);
    await act.waitFor(S.menuDropdown);

    // undo / redo / save are global commands, summoned from the Edit and File submenus rather than docked
    await act.click(S.menuEdit);
    await act.waitFor(S.menuSubmenu);
    assert("edit actions present", (await page.locator(S.menuAction).count()) >= 2);
    await step("edit", { highlight: [S.menuSubmenu] });

    await act.click(S.menuFile);
    await act.waitFor(S.menuSubmenu);
    assert("file actions present", (await page.locator(S.menuAction).count()) >= 1);
    await step("file", { highlight: [S.menuSubmenu] });
});
