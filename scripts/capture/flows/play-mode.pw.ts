import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "play-mode", scene: "demo" }, async ({ page, step, act, assert }) => {
    await step("stopped", { highlight: [S.playBtn] });

    await act.click(S.playBtn);
    await act.wait(1000);

    assert("stop button visible", await page.locator(S.stopBtn).isVisible());
    await step("playing", { highlight: [S.stopBtn] });

    await act.click(S.stopBtn);
    await act.wait(500);

    assert("play button restored", await page.locator(S.playBtn).isVisible());
});
