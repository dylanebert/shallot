import { test } from "playwright/test";
import { capture } from "./capture";

for (const [id, name] of [
    ["joint", "Joint"],
    ["sensor", "Sensor Visit"],
    ["contact", "Contact"],
    ["hit", "Hit"],
    ["motor", "Motor Joint"],
]) {
    test(`${name} event-stage look review`, async ({ page }) => {
        page.on("pageerror", (error) => {
            throw error;
        });
        await page.goto(`/?scene=${id}`);
        await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
        await capture(page, `${id}-default`);
        if (id === "hit" || id === "contact") {
            await page.waitForTimeout(id === "hit" ? 5500 : 1800);
            await capture(page, `${id}-events`);
        }
        if (id === "joint") {
            await page.mouse.move(430, 290);
            await page.mouse.down();
            await page.mouse.move(430, 610, { steps: 10 });
            await page.waitForTimeout(500);
            await capture(page, "joint-grab");
            await page.mouse.up();
        }
        if (id === "motor") {
            await page.getByRole("slider", { name: "Speed", exact: true }).fill("1");
            await page.getByRole("button", { name: "Apply impulse", exact: true }).click();
            await capture(page, "motor-controls");
        }
        if (id === "contact") {
            await page.getByRole("slider", { name: "Torque", exact: true }).fill("45000");
            await page.locator("#canvas").focus();
            await page.keyboard.down("KeyW");
            await capture(page, "contact-input");
            await page.keyboard.up("KeyW");
        }
    });
}
