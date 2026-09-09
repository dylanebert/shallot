import { expect, test } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { verify } from "../../../scripts/verify";
import { chromePoints, frame, isMagenta, requireDisplay } from "./flows/harness";

// By-path headed tier for the sandboxed overlay's bounds, moved here from `examples/flows/ui-containment`
// because the claim is about this package's `config.ui` containment. The fixture mounts a deliberately
// invalid HUD — `position: fixed`, far larger than the canvas — and the engine's sandboxed overlay must
// bound and clip it. Paint containment is not observable from inside the page, so verify captures the
// frame and the pixel assertion runs here (the reason `scripts/flows.ts` held it node-side).
//
//     bun test ./packages/shallot/tests/flow-ui-containment.tier.ts
//
// The canvas is inset 64px from the window, so every chrome sample point must stay clear of the HUD's
// magenta while the window centre — inside the canvas — must carry it, which is what proves the UI
// mounted rather than simply failing to paint.

test("ui-containment: the sandboxed overlay is clipped to the canvas region", async () => {
    requireDisplay();
    const shot = join(tmpdir(), `shallot-flow-ui-${Date.now()}.png`);
    const result = await verify("packages/shallot/tests/flows/ui-containment", [
        "--screenshot",
        shot,
    ]);
    expect(result?.pass).toBe(true);

    const { width, height, at } = frame(shot);
    for (const [x, y] of chromePoints(width, height)) {
        const [r, g, b] = at(x, y);
        expect({ x, y, magenta: isMagenta(r, g, b) }).toEqual({ x, y, magenta: false });
    }
    const [r, g, b] = at(Math.floor(width / 2), Math.floor(height / 2));
    expect({ centre: isMagenta(r, g, b) }).toEqual({ centre: true });
}, 180_000);
