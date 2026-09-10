import {
    GlazePlugin,
    InputPlugin,
    LinesPlugin,
    OrbitPlugin,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    run,
    SearPlugin,
    SlabPlugin,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";

// Shared boot for the visualization demos: the shallot render stack + lines + orbit, the profiler, and
// the HMR wiring. The camera, lights, and content are declared in each demo's `.scene` — a lone
// `<a camera sear orbit transform>` auto-binds to the page's <canvas> and orbits the origin, so the
// single-view case needs no code. Each demo passes the extra plugin(s) its scene references (a feed
// system, AnimationPlugin, TextPlugin) — a plugin's `dependencies` are checked, never auto-added, so a
// demo plugin depending on AnimationPlugin lists it here too or `build` skips the demo plugin with a
// console warning and the scene's `animator` attributes fall through unregistered. F3 toggles the profiler panel (ProfilePlugin); HMR disposes the State.
export async function start(plugins: Plugin[], scene: string) {
    const { dispose } = await run({
        defaults: false,
        scene,
        plugins: [
            ProfilePlugin,
            SlabPlugin,
            TransformsPlugin,
            InputPlugin,
            OrbitPlugin,
            RenderPlugin,
            PartPlugin,
            SearPlugin,
            GlazePlugin,
            LinesPlugin,
            ...plugins,
        ],
    });

    if (import.meta.hot) {
        import.meta.hot.dispose(() => dispose());
    }
}
