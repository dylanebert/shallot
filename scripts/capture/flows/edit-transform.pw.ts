import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "edit-transform", scene: "demo" }, async ({ page, step, act }) => {
    await act.click(page.locator(S.row, { hasText: "ground" }));

    const posField = page
        .locator(`${S.fieldRow}:has(${S.fieldLabel}:text('pos'))`)
        .locator(S.fieldInput)
        .first();
    await act.fill(posField, "2");

    await step("transform-edited", {
        highlight: [`${S.fieldRow}:has(${S.fieldLabel}:text('pos'))`],
        clip: S.inspector,
    });
});
