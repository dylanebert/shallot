import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(1000);

import { sharedGpuBuild } from "../../engine/app/gpu.fixture";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();
const build = await sharedGpuBuild();

test("build awaits an application-owned loading completion promise before cleanup and returning the app", async () => {
    let finish!: () => void;
    let entered!: () => void;
    let cleaned = false;
    let returned = false;
    const completionStarted = new Promise<void>((resolve) => (entered = resolve));
    const built = build({
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
}, 1000);
