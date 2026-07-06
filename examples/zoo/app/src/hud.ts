import { mountOverlay, type Plugin, type State, type System } from "@dylanebert/shallot";

// #doc:intro
// A Shallot project is a scene, a manifest listing the plugins to turn on, and the plugin modules
// themselves. This page is how those fit together, and the two ways a project starts: the `shallot` CLI, or
// `run()` when you embed Shallot in your own bundler.

// #doc:code source:app/public/scenes/app.scene
// ### The scene is the world
//
// The `.scene` file holds the entities — a camera, the lights, a couple of parts. It's plain data the editor
// reads and writes, so most of a project is authored here, not in code. [Scenes](doc:engine/scene) covers
// the format.

// #doc:code source:app/shallot.json
// ### The manifest turns plugins on
//
// `shallot.json` names the scene and the plugins the project enables — `true` for a built-in, a module path
// for one of your own. Run it standalone with `bunx shallot dev`, or open it in the editor with `bunx shallot`.
// Nothing here calls `run()`: the CLI builds the app from this manifest.

// #doc:code
// ### A plugin adds behavior
//
// A plugin is the unit of behavior — a name, the systems it runs, and lifecycle hooks. This one mounts a DOM
// overlay over the canvas and updates it each frame. `mountOverlay` returns a container sandboxed to the
// canvas region, the same surface a `run()` app gets from `config.ui`; the plugin removes it on `dispose`.
// #region hud
let panel: HTMLDivElement | null = null;

const HudSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    update(state: State) {
        if (typeof document === "undefined") return;
        if (!panel) {
            const overlay = mountOverlay(document.querySelector("canvas"));
            panel = document.createElement("div");
            panel.style.cssText =
                "position:absolute;top:12px;left:12px;padding:6px 10px;border-radius:4px;" +
                "background:rgba(14,13,12,0.85);color:#f0ece8;font:12px ui-monospace,monospace";
            overlay.append(panel);
        }
        panel.textContent = `running ${state.time.elapsed.toFixed(1)}s`;
    },
};

const Hud = {
    name: "Hud",
    systems: [HudSystem],
    dispose() {
        panel?.parentElement?.remove();
        panel = null;
    },
} satisfies Plugin;
// #endregion

// #doc:code
// ### Or start it yourself
//
// Embedding Shallot in your own bundler instead of the CLI? Call `run` with the same plugins and scene:
//
// ```ts
// const app = await run({ plugins: [Hud], scene: "/scenes/app.scene" });
// ```
//
// It builds the app and starts the frame loop. `build` returns the app without the loop, for when you drive
// `state.step(dt)` yourself. The `Config` reference below lists every field.

// #doc:code
// While the app builds, Shallot shows the `shallotDark` loading screen — the logo over a progress bar. Pass
// `loading` in your config to switch it: `shallotLight`, or the logo-less `minimalDark` / `minimalLight`.

export default Hud;
