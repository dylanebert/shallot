import { expect, test } from "playwright/test";

test("runApp mounts built UI before completing and removing the loading screen", async ({
    page,
}) => {
    await page.goto("/handoff.html");
    await page.waitForFunction(() => Reflect.get(window, "done"));
    expect(await page.evaluate(() => Reflect.get(window, "events"))).toEqual([
        "show",
        "mount",
        "complete",
        "remove screen",
        "plugin dispose",
        "UI cleanup",
    ]);
    await expect(page.locator("canvas + div")).toHaveCount(0);
});

test("a UI mount failure disposes the build and reports through the loading screen", async ({
    page,
}) => {
    await page.goto("/handoff.html?fail");
    await page.waitForFunction(() => Reflect.get(window, "done"));
    expect(await page.evaluate(() => Reflect.get(window, "events"))).toEqual([
        "show",
        "mount",
        "plugin dispose",
        "error",
        "rejected",
    ]);
    await expect(page.locator("canvas + div")).toHaveCount(0);
});
