import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { runApp } from "./index";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

test("runApp logs a system error once, keeps healthy frames running and resumes after swapSystem", async () => {
    const cause = new Error("broken");
    let attempts = 0;
    let frames = 0;
    let resumed = 0;
    const broken = {
        name: "spawn",
        update() {
            attempts++;
            throw cause;
        },
    };
    const healthy = {
        after: [broken],
        update() {
            frames++;
        },
    };
    const logged = spyOn(console, "error").mockImplementation(() => {});
    const app = await runApp({
        defaults: false,
        plugins: [{ name: "Game", systems: [broken, healthy] }],
    });
    try {
        const waitFor = async (ready: () => boolean) => {
            const deadline = performance.now() + CEILING.node / 2;
            while (!ready()) {
                if (performance.now() > deadline) throw new Error("frame loop did not progress");
                await Bun.sleep(1);
            }
        };
        await waitFor(() => frames >= 3);
        expect(attempts).toBe(1);
        expect(logged).toHaveBeenCalledTimes(1);
        expect(logged.mock.calls[0]).toEqual([
            'System "Game/spawn" threw and is paused until its next reload:',
            cause,
        ]);
        const before = frames;
        app.world.swapSystem(broken, {
            update() {
                resumed++;
            },
        });
        await waitFor(() => frames >= before + 3);
        expect(resumed).toBeGreaterThanOrEqual(3);
        expect(logged).toHaveBeenCalledTimes(1);
    } finally {
        app.dispose();
        logged.mockRestore();
    }
});
