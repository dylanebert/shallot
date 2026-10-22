import { expect, test } from "bun:test";
import { PUBLISH_LAUNCH_OPTIONS } from "./publish-verify";

test("production Chromium enables its sandbox and never passes --no-sandbox", () => {
    expect(PUBLISH_LAUNCH_OPTIONS.chromiumSandbox).toBe(true);
    expect(PUBLISH_LAUNCH_OPTIONS.args).not.toContain("--no-sandbox");
});
