import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { test } from "playwright/test";
import { capture } from "./capture";

test.use(pipelineWarnings);

for (const [id, name] of [
    ["joint", "Joint"],
    ["sensor", "Sensor Visit"],
    ["contact", "Contact"],
    ["hit", "Hit"],
    ["move", "Move"],
    ["persistentContact", "Persistent Contact"],
    ["sensorHits", "Sensor Hits"],
    ["motor", "Motor Joint"],
]) {
    test(`${name} event-stage look review`, async ({ page }) => {
        page.on("pageerror", (error) => {
            throw error;
        });
        await page.goto(`/?scene=${id}`);
        await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
        if (id === "move" || id === "persistentContact" || id === "sensorHits")
            await page.waitForTimeout(1500);
        await capture(page, `${id}-default`);
        if (id === "sensor") {
            await page.waitForTimeout(2000);
            await capture(page, "sensor-reaction");
        }
        if (id === "hit" || id === "contact" || id === "persistentContact") {
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
        if (id === "sensorHits") {
            await page.mouse.move(300, 550);
            await page.mouse.down();
            await page.mouse.move(480, 550, { steps: 12 });
            await page.mouse.up();
            await capture(page, "sensorHits-surfaces");
            await page.getByRole("checkbox", { name: "Bullet", exact: true }).uncheck();
            await page.getByRole("button", { name: "Launch", exact: true }).click();
            await page.waitForTimeout(500);
            await capture(page, "sensorHits-controls");
            await page.setViewportSize({ width: 390, height: 844 });
            await page.goto("/?scene=sensorHits");
            await page.getByRole("button", { name: "Scene", exact: true }).waitFor();
            await page.waitForTimeout(1500);
            await capture(page, "sensorHits-phone-default");
            await page.getByRole("checkbox", { name: "Bullet", exact: true }).uncheck();
            await page.getByRole("button", { name: "Launch", exact: true }).click();
            await page.waitForTimeout(500);
            await capture(page, "sensorHits-phone-controls");
            await page.getByText("Physics", { exact: true }).click();
            await capture(page, "sensorHits-phone-scene");
            await page.mouse.move(90, 650);
            await page.mouse.down();
            await page.mouse.move(190, 650, { steps: 12 });
            await page.mouse.up();
            await capture(page, "sensorHits-phone-surfaces");
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
