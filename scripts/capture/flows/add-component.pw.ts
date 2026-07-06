import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "add-component", scene: "demo" }, async ({ page, step, act, assert }) => {
    await act.click(page.locator(S.row, { hasText: "ground" }));
    await act.click(S.addComponentBtn);
    await act.waitFor(S.addComponentPicker);

    await step("picker-open", {
        highlight: [S.addComponentPicker],
        clip: S.inspector,
    });

    // ArrowDown focuses the first row; keyboard order must follow the visual grouped order
    // (rows are reordered into category buckets), so the focused row is the first rendered one.
    // The full unfiltered list spans multiple categories, where registration order and grouped
    // order genuinely differ — the case the regression hides in.
    await page.keyboard.press("ArrowDown");
    const focused = (await page.locator(`${S.addComponentItem}.focused`).textContent())?.trim();
    const firstRow = (await page.locator(S.addComponentItem).first().textContent())?.trim();
    assert("keyboard focus follows visual order", !!focused && focused === firstRow);

    await act.fill(S.addComponentSearch, "light");

    // the footer tracks the cursor (mouse or keyboard) and shows the focused component's summary
    // plus structured trait chips (lights are singleton / require Transform, so a chip always shows).
    await page.keyboard.press("ArrowDown");
    await act.waitFor(S.addComponentSummary);
    const summary = (await page.locator(S.addComponentSummary).textContent())?.trim() ?? "";
    assert("focused component shows a summary", summary.length > 0);
    const chip = (await page.locator(S.addComponentTag).first().textContent())?.trim() ?? "";
    assert("focused component shows a trait chip", /one per scene|requires|excludes/.test(chip));

    await step("picker-filtered", { clip: S.addComponentPicker });

    await act.click(page.locator(S.addComponentItem).first());

    await step("component-added", { clip: S.inspector });
});
