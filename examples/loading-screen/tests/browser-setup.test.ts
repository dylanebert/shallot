import { expect, test } from "bun:test";
import { assertChromiumInstalled } from "./browser-install";

test("a missing Playwright browser names its native install and test commands", () => {
    expect(() => assertChromiumInstalled("/shallot-browser-install-test/missing-chromium")).toThrow(
        "Run `bunx playwright install chromium`, then `bunx playwright test`.",
    );
});
