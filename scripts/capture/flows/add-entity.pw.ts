import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "add-entity", scene: "demo" }, async ({ page, step, act, assert }) => {
    const beforeCount = await page.locator(S.row).count();

    await step("before", { highlight: [S.addEntityBtn] });
    await act.click(S.addEntityBtn);

    const afterCount = await page.locator(S.row).count();
    assert("new entity appeared", afterCount > beforeCount);

    await step("after", { highlight: [S.outliner] });
});
