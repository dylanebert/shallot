import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, test } from "../../../../scripts/browser.fixture";

test.use(pipelineWarnings);

test("late glTF textures equal warm textures, and only the pending material rebinds after decode", async ({
    page,
}) => {
    await page.goto("/");
    await expect(page.locator("body")).toHaveText(/^\{/);
    const result = JSON.parse(await page.locator("body").innerText());
    expect(result.equal).toBe(true);
    expect(result.textured).toBe(true);
    expect(result.changed).toBe(true);
    expect(result.unchanged).toBe(true);
    // The changed batch owns one color and one depth bind group; the other batch stays cached.
    expect(result.groups).toBe(2);
    expect(result.steady).toBe(0);
});
