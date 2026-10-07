import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../engine";

await setupGlobals();

test("build awaits an application-owned loading completion promise before cleanup and returning the app", async () => {
    let finish!: () => void;
    let entered!: () => void;
    let cleaned = false;
    let returned = false;
    const completionStarted = new Promise<void>((resolve) => (entered = resolve));
    const built = createApp({
        defaults: false,
        plugins: [],
        loading: {
            show: () => () => (cleaned = true),
            update: () => {},
            complete: () => {
                entered();
                return new Promise<void>((resolve) => (finish = resolve));
            },
        },
    }).then((app) => {
        returned = true;
        return app;
    });
    await completionStarted;
    const held = !returned && !cleaned;
    finish();
    const app = await built;
    try {
        expect(held).toBe(true);
        expect(cleaned).toBe(true);
    } finally {
        app.dispose();
    }
});
