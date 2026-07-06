import type { Page } from "@playwright/test";
import { defineFlow } from "../runner";
import { S } from "../selectors";

const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

// the editor-only `Overlays` component is the faithful-preview probe: only GizmosPlugin (editor tooling,
// composed into edit builds, never play) registers it, so a non-empty count means the editor foundation is
// present and an empty one means the build is the app's plugins alone. `find` returns [] for an unregistered
// component, so play (where GizmosPlugin isn't built) reads 0 without erroring.
async function overlayCount(page: Page): Promise<number> {
    const res = await page.request.get(`${API}/entities?component=Overlays`);
    return ((await res.json()) as unknown[]).length;
}

// Play is a faithful preview: the editor builds the app's declared plugins verbatim, none of its own
// foundation composited in (plugins.ts `compose`). So the play viewport runs the scene exactly as it ships
// — no editor overlays, no Scene-View gizmos chrome (Unity's Scene-View-vs-Game-View split). The verbatim
// guarantee itself is unit-tested (`compose("play", app) === app`); this flow proves the live consequence.
defineFlow({ name: "play-mode", scene: "demo" }, async ({ page, step, act, assert, sample }) => {
    // edit composes the editor foundation over the app — the edit camera carries the editor-only Overlays,
    // and the gizmos overlay control is offered in the bar
    const editOverlays = await overlayCount(page);
    assert(`edit composes the overlay foundation (Overlays: ${editOverlays})`, editOverlays >= 1);
    assert(
        "gizmos chrome present in edit",
        await page.getByRole("button", { name: "Gizmos" }).isVisible(),
    );
    await step("stopped", { highlight: [S.playBtn] });

    await act.click(S.playBtn);
    await act.wait(1000);
    assert("stop button visible", await page.locator(S.stopBtn).isVisible());

    // play runs the declared set only: GizmosPlugin (editor tooling) isn't in the build, so nothing carries
    // the editor-only Overlays component and the gizmos chrome is gone from the bar
    const playOverlays = await overlayCount(page);
    assert(
        `play runs the declared set only — no editor overlays (Overlays: ${playOverlays})`,
        playOverlays === 0,
    );
    assert(
        "no gizmos chrome in play",
        !(await page.getByRole("button", { name: "Gizmos" }).isVisible()),
    );

    // the preview still renders the scene as it ships — the fixture declares OrbitPlugin, so its camera poses
    const playing = await sample();
    assert(
        `play renders the scene faithfully (non-background ${playing.nonBackground.toFixed(2)})`,
        playing.nonBackground > 0.3,
    );
    await step("playing", { highlight: [S.stopBtn] });

    await act.click(S.stopBtn);
    await act.wait(500);
    assert("play button restored", await page.locator(S.playBtn).isVisible());
    // stopping returns to edit, and the foundation composes back
    const restoredOverlays = await overlayCount(page);
    assert(
        `edit overlays restored after stop (Overlays: ${restoredOverlays})`,
        restoredOverlays >= 1,
    );
});
