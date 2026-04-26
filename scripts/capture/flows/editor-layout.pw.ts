import { defineFlow } from "../runner";
import { S } from "../selectors";

defineFlow({ name: "editor-layout", scene: "demo" }, async ({ step }) => {
    await step("layout", {
        labels: {
            [S.viewport]: "Viewport",
            [S.outliner]: "Outliner",
            [S.inspector]: "Inspector",
        },
    });
});
