import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { runBrowserCheck } from "./browser";
import { CAPTURE_CONTRACT } from "./capture";

check(
    "the page capture reads the frame presented in its caller task",
    {
        claim: "a capture initiated after one canvas presentation reads a later presentation instead of the frame the caller requested",
        size: "integration",
        requires: ["browser"],
        subject: [
            "src/harness/browser.ts",
            "src/harness/capture.ts",
            "src/harness/fixtures/capture.html",
            "src/harness/fixtures/capture.ts",
        ],
        budget: 20_000,
    },
    () =>
        runBrowserCheck(resolve(import.meta.dir, "fixtures/capture.html"), async (page) => {
            expect(page.viewportSize()).toEqual({
                width: CAPTURE_CONTRACT.width,
                height: CAPTURE_CONTRACT.height,
            });
        }),
);
