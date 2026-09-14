import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { check } from "@dylanebert/shallot/harness/check";

const SERVE = resolve(import.meta.dir, "chromium-serve.fixture.ts");

check(
    "focused canvas keydown reaches the device record in Chromium",
    {
        claim: "a real focused-canvas keydown reaches the State-scoped keyboard record",
        size: "integration",
        requires: ["chromium"],
        subject: ["src/standard/input/index.ts", "src/standard/input/chromium-page.fixture.ts"],
    },
    async () => runBrowserCheck((port) => [process.execPath, SERVE, "--port", String(port)]),
);
