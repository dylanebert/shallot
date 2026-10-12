import { expect, test } from "bun:test";
import { assertChromiumInstalled } from "./browser-install";

test("a missing Playwright browser names its local install command", () => {
    expect(() => assertChromiumInstalled("/shallot-browser-install-test/missing-chromium")).toThrow(
        "Run `bun run browser:install`, then `bun run test:browser`.",
    );
});
