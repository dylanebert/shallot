import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { check } from "@dylanebert/shallot/harness/check";

const SERVE = resolve(import.meta.dir, "chromium-serve.fixture.ts");

check(
    "public Loading profiles hand off through the browser package exports",
    {
        claim: "the public browser Loading row proves ground-first handoff, profile timing, cancellation, errors, progress and reduced motion",
        size: "integration",
        requires: ["chromium"],
        host: "mac",
        subject: [
            "src/brand/index.ts",
            "src/standard/loading/index.ts",
            "src/standard/loading/presentation.ts",
            "src/standard/loading/chromium-page.fixture.ts",
        ],
    },
    async () => runBrowserCheck((port) => [process.execPath, SERVE, "--port", String(port)]),
);
