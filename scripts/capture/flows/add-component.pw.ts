import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "add-component", scene: "demo" }, async ({ page, step, act }) => {
    await act.click(page.locator(S.row, { hasText: "ground" }));
    await act.click(S.addComponentBtn);
    await act.waitFor(S.addComponentPicker);

    await step("picker-open", {
        highlight: [S.addComponentPicker],
        clip: S.inspector,
    });

    await act.fill(S.addComponentSearch, "light");

    await step("picker-filtered", { clip: S.addComponentPicker });

    await act.click(page.locator(S.addComponentItem).first());

    await step("component-added", { clip: S.inspector });
});
