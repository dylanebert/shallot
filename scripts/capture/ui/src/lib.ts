import type { Config } from "@dylanebert/shallot";

// The UI-containment fixture (scripts/capture, ui-containment flow). A standalone run() app whose canvas
// sits in a sub-region with chrome around it (index.html). `config.ui` mounts a deliberately INVALID HUD:
// `position: fixed` (viewport-relative) and far larger than the canvas frame — the exact "spilled over the
// editor" shape the contract forbids. The engine's sandboxed overlay (`contain: layout paint` +
// `overflow: hidden`) must bound + clip it to the canvas region, so the surrounding chrome stays untouched.
// Without that containment the magenta covers the whole window — the flow's chrome-pixel assertion is what
// catches the regression. The magenta here must match MAGENTA in ui-containment.pw.ts.
export const config: Config = {
    plugins: [], // DEFAULT_PLUGINS (camera + sear renderer) fill in — we only need the canvas sized
    scene: "scenes/scene.scene",
    ui: (container) => {
        const bad = document.createElement("div");
        bad.id = "overflow-ui";
        bad.style.cssText =
            "position:fixed;top:0;left:0;width:4000px;height:4000px;pointer-events:none;background:rgb(255,0,255)";
        container.appendChild(bad);
        return () => bad.remove();
    },
};
