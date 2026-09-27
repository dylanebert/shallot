import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { runBrowserCheck } from "./browser";
import { CAPTURE_CONTRACT } from "./capture";

check(
    "the page capture carries its tag at the fixed geometry twice",
    {
        claim: "a browser capture preserves the declared page geometry, shows its color tag, and is byte-identical across two captures of one rendered state",
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
