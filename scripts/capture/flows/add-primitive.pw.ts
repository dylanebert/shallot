import { defineFlow } from "../runner";
import { S } from "../selectors";

// the beginner on-ramp: the outliner `+` → Add menu → Box drops a ready-to-use, visible entity with no
// component search. Asserts the whole contract — the menu, a titled outliner row, the inspector's
// components, the canvas actually rendering it, and a clean undo.
defineFlow(
    { name: "add-primitive", scene: "demo" },
    async ({ page, step, act, assert, sample }) => {
        const before = await page.locator(S.row).count();

        await step("before", { highlight: [S.addEntityBtn] });
        await act.click(S.addEntityBtn);
        await act.waitFor(S.addEntityMenu);
        await step("menu-open", { highlight: [S.addEntityMenu] });

        await act.click(page.locator(S.addEntityItem, { hasText: "Box" }));

        assert("adding Box grows the outliner", (await page.locator(S.row).count()) === before + 1);
        // the demo scene already holds "box", so the mint dedupes to "box-2"
        const row = page.locator(S.row, { hasText: "box-2" });
        assert("the new row is titled box-2", (await row.count()) === 1);
        assert("the new entity is selected", (await page.locator(S.rowSelected).count()) === 1);

        const sections = page.locator(`${S.inspector} ${S.sectionHeader}`);
        assert("inspector shows the bundle's components", (await sections.count()) >= 3);

        const rendered = await sample();
        assert(
            `edit view renders the primitive (non-background ${rendered.nonBackground.toFixed(2)})`,
            rendered.nonBackground > 0.3,
        );
        await step("added", { highlight: [S.outliner, S.inspector] });

        await act.key("Control+z");
        assert("undo removes the primitive", (await page.locator(S.row).count()) === before);
        await step("undone", { highlight: [S.outliner] });
    },
);
