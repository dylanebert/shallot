import { expect } from "bun:test";
import { resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { browserAdapterRefusal, runBrowserCheck } from "./browser";
import { CAPTURE_CONTRACT } from "./capture";

check(
    "the browser seat refuses software adapter identities",
    {
        claim: "a browser requirement refuses SwiftShader and llvmpipe identities instead of counting software rendering as a real device",
    },
    () => {
        expect(
            browserAdapterRefusal({
                present: true,
                info: { vendor: "Google Inc.", device: "Google SwiftShader" },
            }),
        ).toContain("Google Inc. Google SwiftShader");
        expect(
            browserAdapterRefusal({
                present: true,
                info: { vendor: "Mesa", device: "llvmpipe (LLVM 15.0.7, 256 bits)" },
            }),
        ).toContain("Mesa llvmpipe");
        expect(
            browserAdapterRefusal({ present: true, info: { vendor: "Apple", device: "M2" } }),
        ).toBeNull();
    },
);

check(
    "the page capture carries its tag at the fixed geometry twice",
    {
        claim: "a browser capture preserves the declared page geometry, shows its color tag, and is byte-identical across two captures of one rendered state",
        size: "integration",
        requires: ["browser"],
        subject: [
            "src/harness/browser.ts",
            "src/harness/capture.ts",
            "src/harness/pages/capture.html",
            "src/harness/pages/capture.ts",
        ],
        budget: 20_000,
    },
    () =>
        runBrowserCheck(resolve(import.meta.dir, "pages/capture.html"), async (page) => {
            expect(page.viewportSize()).toEqual({
                width: CAPTURE_CONTRACT.width,
                height: CAPTURE_CONTRACT.height,
            });
        }),
);
