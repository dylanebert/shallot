import { mountOverlay, type Plugin, type State, type System } from "@dylanebert/shallot";

// `mountOverlay` returns a container sandboxed to the canvas region — the same surface a `run()` app
// gets from `config.ui`. This plugin mounts a DOM panel into it and updates it each frame. Passing
// `state` ties the overlay's removal to the State's lifetime, so it unwinds at `state.dispose()` — no
// plugin `dispose` hook.
let panel: HTMLDivElement | null = null;

const HudSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    update(state: State) {
        if (typeof document === "undefined") return;
        if (!panel) {
            const overlay = mountOverlay(document.querySelector("canvas"), state);
            panel = document.createElement("div");
            panel.style.cssText =
                "position:absolute;top:12px;left:12px;padding:6px 10px;border-radius:4px;" +
                "background:rgba(14,13,12,0.85);color:#f0ece8;font:12px ui-monospace,monospace";
            overlay.append(panel);
            // clear the module ref on teardown so a rebuilt State re-mounts (the overlay is already gone).
            state.onDispose(() => {
                panel = null;
            });
        }
        panel.textContent = `running ${state.time.elapsed.toFixed(1)}s`;
    },
};

const Hud = {
    name: "Hud",
    systems: [HudSystem],
} satisfies Plugin;

// embedding Shallot in your own bundler rather than the CLI: call `run` with the same plugins and scene
//
// ```ts
// const app = await run({ plugins: [Hud], scene: "/scenes/overlay-ui.scene" });
// ```
//
// `run` builds the app and starts the frame loop; `build` returns the app without the loop, for driving
// `state.step(dt)` yourself. While it builds, Shallot shows the `shallotDark` loading screen — pass
// `loading` in your config for `shallotLight` or the logo-less `minimalDark`/`minimalLight`.
export default Hud;
