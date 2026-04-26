import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "select-entity", scene: "demo" }, async ({ page, step, act, assert }) => {
    await act.click(page.locator(S.row, { hasText: "ground" }));

    await step("selected", {
        click: `${S.row}:has(${S.label}:text('ground'))`,
        highlight: [S.inspector],
    });

    const sections = page.locator(`${S.inspector} ${S.sectionHeader}`);
    assert("inspector shows components", (await sections.count()) > 0);
});
