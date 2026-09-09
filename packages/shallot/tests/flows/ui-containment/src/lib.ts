import type { Config } from "@dylanebert/shallot";

// The ui-containment flow's standalone app. Its canvas sits in a sub-region with host chrome around it
// (index.html). `config.ui` mounts a deliberately INVALID HUD: `position: fixed` (viewport-relative) and far
// larger than the canvas frame — the exact "spilled over the host" shape the contract forbids. The engine's
// sandboxed overlay (`contain: layout paint` + `overflow: hidden`) must bound + clip it to the canvas region
// so the surrounding chrome stays untouched. Without that containment the magenta covers the whole window —
// the flow's chrome-pixel assertion (scripts/flows.ts) is what catches the regression. The magenta here must
// match MAGENTA in scripts/flows.ts.
// inline scene, never a public/scenes file: a .scene on disk makes `shallot verify` classify the dir as a
// manifest project, whose synthesized entry bypasses this app's index.html + main.ts.
const scene = `<scene>
    <a ambient-light="intensity: 0.6" />
    <a camera sear transform />
</scene>`;

export const config: Config = {
    plugins: [], // DEFAULT_PLUGINS (camera + sear renderer) fill in — we only need the canvas sized
    scene,
    ui: (container) => {
        const bad = document.createElement("div");
        bad.id = "overflow-ui";
        bad.style.cssText =
            "position:fixed;top:0;left:0;width:4000px;height:4000px;pointer-events:none;background:rgb(255,0,255)";
        container.appendChild(bad);
        return () => bad.remove();
    },
};
