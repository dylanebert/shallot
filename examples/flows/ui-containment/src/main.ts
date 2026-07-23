import { run } from "@dylanebert/shallot";
import { installHarness } from "@dylanebert/shallot/harness";
import { config } from "./lib";

// Boot the standalone run() app and install a minimal published harness: run() reports the mount succeeded.
// `shallot verify --screenshot` drives it and captures the frame; the pixel containment assertion lives in
// scripts/flows.ts (paint containment isn't observable in-page).
const { state, dispose } = await run(config);
const harness = installHarness(state);
// the canvas renders only a solid-magenta HUD fill — a flat clear color, not framed structure. The
// pixel-containment assertion is node-side (scripts/flows.ts reads the screenshot), so opt this harness
// out of verify's structure gate; it would otherwise read the flat fill as a blank canvas and fail.
harness.noRender = true;
harness.run = async () => ({ ok: true, checks: [{ name: "ui mounted", ok: true }] });

// HMR re-runs this module — dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => dispose());
}
